import json

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from support_router.common import hardware, read_json, run_paths, sha, write_json
from support_router.data import normalize, parse_raw
from support_router.models import benchmark, predict_proba, stable_top
from support_router.policy import policy_stats, route
from support_router.runtime import Router
from support_router.training import metrics

matplotlib.use("Agg")


def evaluate(config, run_id):
    from pathlib import Path

    import matplotlib.pyplot as plt

    run_id, artifact, report = run_paths(config, run_id)
    if (report / "test_metrics.json").exists():
        saved = read_json(report / "test_metrics.json")
        if saved["manifest_sha256"] != sha(artifact / "manifest.json"):
            raise ValueError("Test has already been viewed for a different manifest")
        print("Final evaluation already saved; no model or threshold retuning", flush=True)
        return
    router = Router(config, run_id)
    source = read_json(artifact / "source.json")
    path = Path(config["project"]["data_dir"]) / source["datasets"]["test"]["path"]
    test = parse_raw(path.read_bytes(), 3080)
    if set(test.category) != set(router.classes):
        raise ValueError("Official test categories differ from frozen training labels")
    test["sample_id"] = [f"test-{i:05d}" for i in range(len(test))]
    texts = test.text.map(normalize)
    probabilities = predict_proba(router.model, router.family, texts, router.encoder)
    predictions = np.asarray(router.classes)[probabilities.argmax(axis=1)]
    result = {
        "run_id": run_id,
        "mode": "full",
        "split": "official_test",
        "manifest_sha256": sha(artifact / "manifest.json"),
        "selection_sha256": sha(artifact / "selection.json"),
        "policy_sha256": sha(artifact / "policy.json"),
        "model": router.selection["winner"]["id"],
        **metrics(probabilities, test.category, router.classes),
        "routing": policy_stats(probabilities, test.category, router.classes, router.policy),
    }
    pd.DataFrame(
        classification_report(
            test.category, predictions, labels=router.classes, output_dict=True, zero_division=0
        )
    ).T.to_csv(report / "per_class.csv")
    matrix = confusion_matrix(test.category, predictions, labels=router.classes)
    pd.DataFrame(matrix, index=router.classes, columns=router.classes).to_csv(
        report / "confusion_matrix.csv"
    )
    pairs = [
        {
            "true_category": router.classes[i],
            "predicted_category": router.classes[j],
            "count": int(matrix[i, j]),
        }
        for i in range(77)
        for j in range(77)
        if i != j and matrix[i, j]
    ]
    pairs.sort(key=lambda row: (-row["count"], row["true_category"], row["predicted_category"]))
    pd.DataFrame(pairs, columns=["true_category", "predicted_category", "count"]).to_csv(
        report / "top_confusions.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    shown = pairs[:20][::-1]
    ax.barh(
        [p["true_category"] + " → " + p["predicted_category"] for p in shown],
        [p["count"] for p in shown],
    )
    ax.set_xlabel("Official test errors (frozen champion)")
    fig.tight_layout()
    fig.savefig(report / "top_confusions.png", dpi=140)
    plt.close(fig)
    errors = []
    for i in np.flatnonzero(predictions != test.category.to_numpy()):
        row = test.iloc[i]
        top = stable_top(probabilities[i], router.classes, 3)
        decision, _ = route(top[0]["score"], router.policy)
        errors.append(
            {
                "sample_id": row.sample_id,
                "text": row.text,
                "true_category": row.category,
                "predicted_category": predictions[i],
                "confidence": top[0]["score"],
                "top3": json.dumps(top),
                "routing_decision": decision,
            }
        )
    errors_frame = pd.DataFrame(
        errors,
        columns=[
            "sample_id",
            "text",
            "true_category",
            "predicted_category",
            "confidence",
            "top3",
            "routing_decision",
        ],
    ).sort_values("confidence", ascending=False)
    errors_frame.to_csv(report / "errors.csv", index=False)
    errors_frame.head(20).to_csv(report / "error_review_sample.csv", index=False)
    (report / "error_review.md").write_text(
        "# Автоматическая подборка ошибок\n\n"
        "error_review_sample.csv: до 20 самых уверенных ошибок официального test.\n"
        "Это автоматическая подборка, не ручной анализ студента. Сравните формулировки,\n"
        "близкие категории и похожие train_fit примеры; запишите собственные выводы.\n",
        encoding="utf-8",
    )
    classification = benchmark(lambda t: router.predict(t, include_similar=False), texts)
    full = benchmark(lambda t: router.predict(t, include_similar=True), texts)
    classification["retrieval_included"] = False
    full["retrieval_included"] = True
    timing = {
        "run_id": run_id,
        "split": "official_test",
        "sample_ids": test.sample_id.tolist()[:100],
        "classification": classification,
        "full_response": full,
        "threads": config["model"]["threads"],
        "hardware": hardware(),
        "versions": router.manifest["versions"],
        "artifact_total_bytes": sum(p.stat().st_size for p in artifact.iterdir() if p.is_file()),
        "encoder_bytes": sum(
            info["bytes"] for name, info in source["files"].items() if name.startswith("resources/")
        ),
    }
    write_json(report / "latency.json", timing)
    write_json(
        report / "example_response.json",
        router.predict("My card payment appears twice in the app."),
    )
    write_json(report / "test_metrics.json", result)
    print(
        f"Official test macro-F1={result['macro_f1']:.4f}; accuracy={result['accuracy']:.4f}",
        flush=True,
    )
