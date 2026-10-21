import platform
import time
import warnings

import joblib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from support_router.common import (
    PREPROCESSING,
    check_file,
    file_info,
    hardware,
    now,
    read_json,
    run_paths,
    sha,
    versions,
    write_json,
)
from support_router.data import load_split, verify_source
from support_router.models import (
    baseline,
    benchmark,
    cached_embeddings,
    encoder_metadata,
    load_encoder,
    logistic,
    predict_proba,
    save_model,
)
from support_router.policy import choose_policy


def metrics(probabilities, truth, classes):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    classes = np.asarray(classes)
    predictions = classes[probabilities.argmax(axis=1)]
    top3 = classes[np.argsort(-probabilities, axis=1, kind="stable")[:, :3]]
    return {
        "n": len(truth),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, labels=classes, average="macro")),
        "weighted_f1": float(f1_score(truth, predictions, labels=classes, average="weighted")),
        "top3_accuracy": float((top3 == np.asarray(truth)[:, None]).any(axis=1).mean()),
    }


def select_candidate(rows, config):
    best = max(row["metrics"]["macro_f1"] for row in rows)
    candidates = [r for r in rows if best - r["metrics"]["macro_f1"] <= config["model"]["tie_f1"]]
    fastest = min(r["latency"]["median_ms"] for r in candidates)
    tied = [
        r
        for r in candidates
        if r["latency"]["median_ms"]
        <= fastest * (1 + config["model"]["latency_indistinguishable_ratio"])
    ]
    return min(tied, key=lambda r: (r["family"] != "tfidf", r["C"]))


def train(config, run_id):
    run_id, artifact, report = run_paths(config, run_id)
    if (artifact / "selection.json").exists():
        selection = read_json(artifact / "selection.json")
        for row in selection["candidates"]:
            check_file(artifact / row["file"], row["artifact"])
        print("Training already complete; selection remains frozen", flush=True)
        return
    fit = load_split(config, run_id, "train_fit")
    validation = load_split(config, run_id, "validation_model")
    source = read_json(artifact / "source.json")
    from pathlib import Path

    verify_source(Path(config["project"]["data_dir"]), source)
    encoder = load_encoder(config, source)
    metadata = encoder_metadata(encoder, source)
    write_json(artifact / "encoder.json", metadata)
    embeddings = {
        "train_fit": cached_embeddings(config, encoder, fit, metadata),
        "validation_model": cached_embeddings(config, encoder, validation, metadata),
    }
    rows = []
    for family in ["tfidf", "minilm"]:
        for c in config["model"]["c_values"]:
            candidate_id = f"{family}-C{c:g}"
            path = artifact / f"{candidate_id}.joblib"
            checkpoint = artifact / f"{candidate_id}.json"
            if checkpoint.exists():
                row = read_json(checkpoint)
                check_file(path, row["artifact"])
                rows.append(row)
                continue
            print(f"Training {candidate_id}", flush=True)
            model = baseline(c, config) if family == "tfidf" else logistic(c, config)
            inputs = fit.normalized.tolist() if family == "tfidf" else embeddings["train_fit"]
            start = time.perf_counter()
            # Fail rather than accepting a silently non-converged candidate.
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(inputs, fit.category)
            seconds = time.perf_counter() - start
            probabilities = model.predict_proba(
                validation.normalized.tolist()
                if family == "tfidf"
                else embeddings["validation_model"]
            )
            save_model(path, model)
            restored = joblib.load(path)
            import numpy as np

            if not np.allclose(
                probabilities,
                predict_proba(restored, family, validation.normalized, encoder),
                atol=1e-6,
            ):
                raise ValueError("Serialization changed predictions")
            latency = benchmark(
                lambda text: predict_proba(model, family, [text], encoder), validation.normalized
            )
            latency.update(
                {
                    "retrieval_included": False,
                    "split": "validation_model",
                    "sample_ids": validation.sample_id.tolist()[:100],
                }
            )
            row = {
                "id": candidate_id,
                "family": family,
                "C": c,
                "file": path.name,
                "artifact": file_info(path),
                "metrics": metrics(probabilities, validation.category, model.classes_),
                "training_seconds": seconds,
                "latency": latency,
                "converged": True,
                "iterations": (model[-1] if family == "tfidf" else model).n_iter_.tolist(),
            }
            write_json(checkpoint, row)
            rows.append(row)
            print(f"{candidate_id}: macro-F1={row['metrics']['macro_f1']:.4f}", flush=True)
    winner = select_candidate(rows, config)
    model = joblib.load(artifact / winner["file"])
    selection = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": now(),
        "selected_before_test": True,
        "selection_split": "validation_model",
        "rule": {
            "macro_f1_tolerance": config["model"]["tie_f1"],
            "latency_tie_ratio": config["model"]["latency_indistinguishable_ratio"],
            "tie_preference": "tfidf_then_smaller_C",
        },
        "winner": winner,
        "candidates": rows,
        "labels": model.classes_.tolist(),
        "split_manifest_hash": sha(artifact / "splits.json"),
        "source_manifest_hash": sha(artifact / "source.json"),
        "encoder": metadata,
        "preprocessing": PREPROCESSING,
        "python": platform.python_version(),
        "versions": versions(),
        "hardware": hardware(),
        "threads": config["model"]["threads"],
    }
    write_json(artifact / "selection.json", selection)
    write_json(report / "validation_results.json", selection)
    pd.DataFrame(
        [
            {
                "id": r["id"],
                **r["metrics"],
                "training_seconds": r["training_seconds"],
                "artifact_bytes": r["artifact"]["bytes"],
                "median_ms": r["latency"]["median_ms"],
                "p95_ms": r["latency"]["p95_ms"],
            }
            for r in rows
        ]
    ).to_csv(report / "validation_results.csv", index=False)
    print(f"Frozen champion: {winner['id']}", flush=True)


def tune_policy(config, run_id):
    run_id, artifact, report = run_paths(config, run_id)
    selection = read_json(artifact / "selection.json")
    winner = selection["winner"]
    check_file(artifact / winner["file"], winner["artifact"])
    if (artifact / "policy.json").exists():
        policy = read_json(artifact / "policy.json")
        if policy["classifier_sha256"] != winner["artifact"]["sha256"]:
            raise ValueError("Policy/model mismatch")
        print("Policy already frozen", flush=True)
        return
    frame = load_split(config, run_id, "validation_policy")
    model = joblib.load(artifact / winner["file"])
    encoder = load_encoder(config, read_json(artifact / "source.json"))
    probabilities = predict_proba(model, winner["family"], frame.normalized, encoder)
    policy, table = choose_policy(
        probabilities,
        frame.category,
        model.classes_,
        config["policy"]["target_accuracy"],
        config["policy"]["min_accepted"],
    )
    policy.update(
        {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": now(),
            "classifier_sha256": winner["artifact"]["sha256"],
            "selection_sha256": sha(artifact / "selection.json"),
            "split_manifest_sha256": sha(artifact / "splits.json"),
        }
    )
    write_json(artifact / "policy.json", policy)
    write_json(report / "policy.json", policy)
    pd.DataFrame(table).to_csv(report / "policy_thresholds.csv", index=False)
    print(f"Policy frozen: threshold={policy['threshold']}", flush=True)
