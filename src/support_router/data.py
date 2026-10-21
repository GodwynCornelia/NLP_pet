import io
import re
import time
import unicodedata
from pathlib import Path

import httpx
import pandas as pd
from huggingface_hub import HfApi, snapshot_download
from sklearn.model_selection import train_test_split

from support_router.common import (
    PREPROCESSING,
    check_file,
    digest,
    file_info,
    now,
    read_json,
    run_paths,
    write_json,
)
from support_router.config import load_resource_lock, portable_config


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def duplicate_key(text):
    return normalize(text).casefold()


def request(url):
    for attempt in range(3):
        try:
            response = httpx.get(url, timeout=60, follow_redirects=True)
            response.raise_for_status()
            return response
        except httpx.HTTPError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def parse_raw(content, expected_rows=None):
    frame = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False, encoding="utf-8")
    if list(frame.columns) != ["text", "category"]:
        raise ValueError("BANKING77 schema must be text,category; HTML/error pages rejected")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"Unexpected CSV record count: {len(frame)}, expected {expected_rows}")
    for col in ["text", "category"]:
        if not frame[col].map(lambda x: isinstance(x, str) and bool(normalize(x))).all():
            raise ValueError(f"Invalid empty/missing {col}")
    if frame.category.nunique() != 77:
        raise ValueError("BANKING77 requires exactly 77 categories")
    return frame


def verify_source(root, source):
    for relative, info in source["files"].items():
        check_file(root / relative, info)


def download(config, refresh=False):
    root = Path(config["project"]["data_dir"])
    pointer = root / "source.json"
    lock = None if refresh else load_resource_lock(config)
    if pointer.exists() and not refresh:
        source = read_json(pointer)
        verify_source(root, source)
        if (
            source["dataset_commit"] != lock["dataset_commit"]
            or source["encoder"]["revision"] != lock["encoder_revision"]
        ):
            raise ValueError("local_source_differs_from_resource_lock; use an isolated copy")
        verify_source(root, lock)
        print("Resources verified from local cache", flush=True)
        return source
    if lock is not None:
        revision = lock["dataset_commit"]
    else:
        try:
            revision = request(
                "https://api.github.com/repos/PolyAI-LDN/task-specific-datasets/commits/master"
            ).json()["sha"]
        except (httpx.HTTPError, KeyError):
            revision = None
    raw = root / "raw" / (revision or now().replace(":", "-"))
    raw.mkdir(parents=True, exist_ok=True)
    files, datasets = {}, {}
    for split, size in [("train", 10003), ("test", 3080)]:
        url = (
            "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
            f"{revision or 'master'}/banking_data/{split}.csv"
        )
        path = raw / f"{split}.csv"
        rel = path.relative_to(root).as_posix()
        if path.exists():
            if lock is not None:
                check_file(path, lock["files"][rel])
            content = path.read_bytes()
        else:
            content = request(url).content
        frame = parse_raw(content, size)
        if not path.exists():
            temporary = path.with_suffix(".csv.tmp")
            temporary.write_bytes(content)
            if lock is not None:
                check_file(temporary, lock["files"][rel])
            temporary.replace(path)
        files[rel] = file_info(path)
        # Only training label distribution is audited before selection.
        datasets[split] = {"url": url, "path": rel, "rows": len(frame), "license": "CC-BY-4.0"}
    encoder_id = config["model"]["encoder_id"]
    encoder_revision = (
        lock["encoder_revision"]
        if lock is not None
        else HfApi().model_info(encoder_id, timeout=60, token=False).sha
    )
    encoder = root / "resources" / encoder_revision
    print(f"Downloading CPU encoder snapshot {encoder_revision}", flush=True)
    snapshot_download(
        encoder_id,
        revision=encoder_revision,
        local_dir=encoder,
        allow_patterns=(
            [
                name.removeprefix(f"resources/{encoder_revision}/")
                for name in lock["files"]
                if name.startswith("resources/")
            ]
            if lock is not None
            else ["*.json", "*.txt", "*.safetensors", "1_Pooling/*", "README.md"]
        ),
        max_workers=2,
        token=False,
    )
    for path in sorted(encoder.rglob("*")):
        if path.is_file() and ".cache" not in path.relative_to(encoder).parts:
            files[path.relative_to(root).as_posix()] = file_info(path)
    if lock is not None:
        verify_source(root, lock)
    source = {
        "schema_version": 1,
        "created_at": now(),
        "dataset_commit": revision,
        "commit_pinned": revision is not None,
        "datasets": datasets,
        "encoder": {
            "id": encoder_id,
            "revision": encoder_revision,
            "path": encoder.relative_to(root).as_posix(),
            "license": "Apache-2.0",
        },
        "files": files,
    }
    write_json(root / "manifests" / f"{digest(source)}.json", source)
    write_json(pointer, source)
    return source


def clean_training(train, test_texts):
    train = train.copy()
    train["normalized"] = train.text.map(normalize)
    train["group_id"] = train.text.map(lambda x: digest(duplicate_key(x)))
    test_keys = {digest(duplicate_key(t)) for t in test_texts}
    conflicts = set(train.groupby("group_id").category.nunique().loc[lambda x: x > 1].index)
    excluded, retained, seen = [], [], set()
    for row in train.to_dict("records"):
        key = row["group_id"]
        reason = (
            "conflicting_labels"
            if key in conflicts
            else "official_test_overlap"
            if key in test_keys
            else "duplicate_same_label"
            if key in seen
            else None
        )
        if reason:
            excluded.append(row | {"reason": reason})
        else:
            retained.append(row)
            seen.add(key)
    return pd.DataFrame(retained), pd.DataFrame(excluded)


def split_training(clean, seed=42, expected_classes=77):
    fit, held = train_test_split(clean, test_size=0.30, stratify=clean.category, random_state=seed)
    model, policy = train_test_split(
        held, test_size=0.50, stratify=held.category, random_state=seed
    )
    splits = {"train_fit": fit, "validation_model": model, "validation_policy": policy}
    for name, frame in splits.items():
        if frame.category.nunique() != expected_classes:
            raise ValueError(f"Insufficient class coverage in {name}")
    groups = [set(frame.group_id) for frame in splits.values()]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Duplicate leakage between splits")
    return {k: v.sort_values("sample_id").reset_index(drop=True) for k, v in splits.items()}


def prepare(config, run_id=None):
    root = Path(config["project"]["data_dir"])
    source = read_json(root / "source.json")
    verify_source(root, source)
    fingerprint = digest(
        {"source": source, "config": portable_config(config), "preprocessing": PREPROCESSING}
    )
    if run_id is None:
        pointer = Path(config["project"]["artifact_dir"]) / "current.json"
        if pointer.exists():
            existing = read_json(pointer)["run_id"]
            old = read_json(run_paths(config, existing)[1] / "run.json")
            if old["fingerprint"] == fingerprint:
                print(f"Reusing prepared run {existing}", flush=True)
                return existing
        run_id = now().replace(":", "-").replace("+00-00", "Z") + "-" + fingerprint[:8]
    run_id, artifact, report = run_paths(config, run_id)
    if artifact.exists():
        raise ValueError("Run already exists; use its saved split without re-preparing")
    train = parse_raw((root / source["datasets"]["train"]["path"]).read_bytes(), 10003)
    # Do not read test labels in preprocessing; only its texts are used for overlap exclusion.
    test_texts = pd.read_csv(
        root / source["datasets"]["test"]["path"],
        usecols=["text"],
        dtype=str,
        keep_default_na=False,
    ).text
    train["sample_id"] = [f"train-{i:05d}" for i in range(len(train))]
    clean, excluded = clean_training(train, test_texts)
    splits = split_training(clean, config["project"]["seed"])
    processed = root / "processed" / run_id
    processed.mkdir(parents=True)
    artifact.mkdir(parents=True)
    report.mkdir(parents=True)
    split_manifest = {
        "seed": config["project"]["seed"],
        "preprocessing": PREPROCESSING,
        "source_manifest_hash": digest(source),
        "splits": {},
    }
    for name, frame in splits.items():
        path = processed / f"{name}.csv"
        frame.to_csv(path, index=False)
        split_manifest["splits"][name] = {
            "sample_ids": frame.sample_id.tolist(),
            "group_ids": frame.group_id.tolist(),
            "assignment_hash": digest(frame[["sample_id", "group_id", "category"]].values.tolist()),
            "file": file_info(path),
            "rows": len(frame),
            "class_counts": frame.category.value_counts().sort_index().to_dict(),
        }
    excluded.to_csv(report / "excluded_training_rows.csv", index=False)
    write_json(artifact / "source.json", source)
    write_json(artifact / "splits.json", split_manifest)
    write_json(artifact / "config.json", portable_config(config))
    write_json(
        artifact / "run.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": now(),
            "fingerprint": fingerprint,
            "mode": "full",
            "config_hash": digest(portable_config(config)),
        },
    )
    write_json(
        report / "data_audit.json",
        {
            "run_id": run_id,
            "original_train": len(train),
            "clean_train": len(clean),
            "excluded": len(excluded),
            "exclusion_reasons": excluded.reason.value_counts().to_dict() if len(excluded) else {},
            "raw_train_class_counts": train.category.value_counts().to_dict(),
            "text_length_characters": train.text.str.len().describe().to_dict(),
            "empty_texts": int((train.text.str.strip() == "").sum()),
            "missing_values": 0,
            "test_unchanged": True,
            "test_labels_used_for_cleanup": False,
            "splits": {k: len(v) for k, v in splits.items()},
        },
    )
    write_json(Path(config["project"]["artifact_dir"]) / "current.json", {"run_id": run_id})
    print(f"Prepared {run_id}: {len(clean)} clean training records", flush=True)
    return run_id


def load_split(config, run_id, name):
    if name not in {"train_fit", "validation_model", "validation_policy"}:
        raise ValueError("Training functions cannot load official test")
    _, artifact, _ = run_paths(config, run_id)
    manifest = read_json(artifact / "splits.json")
    path = Path(config["project"]["data_dir"]) / "processed" / run_id / f"{name}.csv"
    check_file(path, manifest["splits"][name]["file"])
    return pd.read_csv(path, dtype=str, keep_default_na=False)
