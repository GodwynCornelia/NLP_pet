from pathlib import Path

import numpy as np
import pandas as pd

from support_router.common import (
    check_file,
    digest,
    file_info,
    now,
    read_json,
    run_paths,
    sha,
    versions,
    write_json,
)
from support_router.data import duplicate_key, load_split
from support_router.models import cached_embeddings, load_encoder


def build_index(config, run_id):
    run_id, artifact, _ = run_paths(config, run_id)
    if (artifact / "manifest.json").exists():
        from support_router.runtime import verify_manifest

        verify_manifest(config, run_id)
        print("Index and manifest already verified", flush=True)
        return
    selection = read_json(artifact / "selection.json")
    policy = read_json(artifact / "policy.json")
    source = read_json(artifact / "source.json")
    fit = load_split(config, run_id, "train_fit")
    encoder = load_encoder(config, source)
    matrix = cached_embeddings(config, encoder, fit, selection["encoder"])
    np.save(artifact / "index.npy", matrix, allow_pickle=False)
    fit[["sample_id", "text", "category", "group_id"]].to_csv(artifact / "index.csv", index=False)
    index = {
        "run_id": run_id,
        "source_split": "train_fit",
        "rows": len(fit),
        "sample_ids": fit.sample_id.tolist(),
        "encoder_revision": source["encoder"]["revision"],
        "split_manifest_sha256": sha(artifact / "splits.json"),
        "dimension": matrix.shape[1],
        "dtype": str(matrix.dtype),
        "files": {f: file_info(artifact / f) for f in ["index.npy", "index.csv"]},
    }
    write_json(artifact / "index.json", index)
    if policy["classifier_sha256"] != selection["winner"]["artifact"]["sha256"]:
        raise ValueError("Policy/model mismatch")
    required = [
        "run.json",
        "config.json",
        "source.json",
        "splits.json",
        "selection.json",
        "policy.json",
        "encoder.json",
        "index.json",
        "index.npy",
        "index.csv",
        selection["winner"]["file"],
    ]
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": now(),
        "seed": config["project"]["seed"],
        "config_hash": read_json(artifact / "run.json")["config_hash"],
        "source_manifest_hash": sha(artifact / "source.json"),
        "split_manifest_hash": sha(artifact / "splits.json"),
        "labels": selection["labels"],
        "classifier_classes": selection["labels"],
        "classifier": selection["winner"],
        "encoder": selection["encoder"],
        "preprocessing": selection["preprocessing"],
        "train_fit_sample_ids_hash": digest(fit.sample_id.tolist()),
        "policy_hash": sha(artifact / "policy.json"),
        "index_hash": sha(artifact / "index.json"),
        "versions": versions(),
        "python": selection["python"],
        "files": {f: file_info(artifact / f) for f in required},
    }
    write_json(artifact / "manifest.json", manifest)
    print(f"Built index: {len(fit)} train_fit examples", flush=True)


class RetrievalIndex:
    def __init__(self, artifact, metadata, expected_ids):
        artifact = Path(artifact)
        for file, info in metadata["files"].items():
            check_file(artifact / file, info)
        self.matrix = np.load(artifact / "index.npy", allow_pickle=False)
        self.rows = pd.read_csv(artifact / "index.csv", dtype=str, keep_default_na=False)
        if metadata["source_split"] != "train_fit" or self.rows.sample_id.tolist() != expected_ids:
            raise ValueError("index_split_mismatch")
        if metadata["sample_ids"] != expected_ids:
            raise ValueError("index_manifest_ids_mismatch")
        if self.matrix.shape != (len(self.rows), metadata["dimension"]):
            raise ValueError("index_shape_mismatch")
        if self.matrix.dtype != np.float32 or not np.isfinite(self.matrix).all():
            raise ValueError("invalid_index_dtype_or_values")
        if not np.allclose(np.linalg.norm(self.matrix, axis=1), 1, atol=1e-4):
            raise ValueError("index_not_normalized")

    def search(self, vector, text, k=3):
        similarities = np.clip(self.matrix @ vector, -1, 1)
        indices = np.argsort(-similarities, kind="stable")[:k]
        result = []
        for index in indices:
            row = self.rows.iloc[index]
            result.append(
                {
                    "sample_id": row.sample_id,
                    "text": row.text,
                    "category": row.category,
                    "similarity": float(similarities[index]),
                    "exact_match": duplicate_key(row.text) == duplicate_key(text),
                }
            )
        return result
