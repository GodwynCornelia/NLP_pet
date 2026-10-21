"""Export compact, allowlisted evidence from saved runs without fitting models."""

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from support_router.common import read_json, sha
from support_router.config import ROOT

LATENCY_KEYS = (
    "median_ms",
    "p95_ms",
    "n",
    "warmup",
    "batch_size",
    "http_included",
    "network",
    "retrieval_included",
)
HARDWARE_KEYS = (
    "cpu",
    "platform",
    "python",
    "physical_cores",
    "logical_cores",
    "ram_bytes",
    "available_ram_bytes",
    "process_rss_bytes",
)
POLICY_KEYS = (
    "schema_version",
    "run_id",
    "created_at",
    "auto_routing_enabled",
    "threshold",
    "target_accuracy",
    "min_accepted",
    "selection_split",
    "n",
    "validation_metrics",
    "classifier_sha256",
    "selection_sha256",
    "split_manifest_sha256",
)
TEST_KEYS = (
    "run_id",
    "mode",
    "split",
    "model",
    "n",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "top3_accuracy",
    "routing",
    "manifest_sha256",
    "selection_sha256",
    "policy_sha256",
)


def pick(value, keys):
    return {key: value[key] for key in keys}


def write_json(path, value):
    """Keep published hashes stable on Windows and Linux checkouts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def public_selection(selection):
    """Do not copy encoder tokenizer reprs: they contain machine-specific paths."""
    return {
        **pick(selection, ("run_id", "selection_split", "selected_before_test", "rule")),
        "winner": selection["winner"]["id"],
        "candidates": [
            {
                **pick(row, ("id", "family", "C", "metrics", "training_seconds", "converged")),
                "latency": pick(row["latency"], LATENCY_KEYS + ("split",)),
            }
            for row in selection["candidates"]
        ],
    }


def decision_counts(test, matrix, errors):
    n, correct = int(matrix.sum()), int(matrix.trace())
    if n != test["n"] or len(errors) != n - correct:
        raise ValueError("test_count_mismatch")
    if not math.isclose(correct / n, test["accuracy"], abs_tol=1e-12):
        raise ValueError("test_accuracy_mismatch")
    top3 = correct + sum(
        row.true_category in [p["category"] for p in json.loads(row.top3)]
        for row in errors.itertuples()
    )
    if not math.isclose(top3 / n, test["top3_accuracy"], abs_tol=1e-12):
        raise ValueError("test_top3_mismatch")
    route = test["routing"]
    accepted, accepted_correct = route["accepted_count"], route["accepted_correct"]
    automatic_errors = int((errors.routing_decision == "auto_route").sum())
    if automatic_errors != accepted - accepted_correct:
        raise ValueError("routing_error_count_mismatch")
    return {
        "all": {"total": n, "correct": correct, "errors": n - correct},
        "auto_route": {
            "total": accepted,
            "correct": accepted_correct,
            "errors": automatic_errors,
        },
        "manual_review": {
            "total": n - accepted,
            "correct": correct - accepted_correct,
            "errors": len(errors) - automatic_errors,
        },
        "top3_correct": top3,
    }


def export_run(root=ROOT, run_id=None):
    root = Path(root)
    run_id = run_id or read_json(root / "artifacts/current.json")["run_id"]
    artifact, report = root / "artifacts" / run_id, root / "reports" / run_id
    output = root / "reports/public"
    selection = read_json(artifact / "selection.json")
    test = read_json(report / "test_metrics.json")
    if test["mode"] != "full" or test["manifest_sha256"] != sha(artifact / "manifest.json"):
        raise ValueError("matching_completed_full_benchmark_required")
    assert test["selection_sha256"] == sha(artifact / "selection.json")
    assert test["policy_sha256"] == sha(artifact / "policy.json")
    audit = read_json(report / "data_audit.json")
    source = read_json(artifact / "source.json")
    splits = read_json(artifact / "splits.json")
    timing = read_json(report / "latency.json")
    policy = read_json(artifact / "policy.json")
    matrix = pd.read_csv(report / "confusion_matrix.csv", index_col=0).to_numpy()
    errors = pd.read_csv(report / "errors.csv")
    counts = decision_counts(test, matrix, errors)
    public = {
        "benchmark.json": {**pick(test, TEST_KEYS), "decision_counts": counts},
        "selection.json": public_selection(selection),
        "policy.json": pick(policy, POLICY_KEYS),
        "latency.json": {
            **pick(timing, ("run_id", "split", "threads", "versions", "sample_ids")),
            "hardware": pick(timing["hardware"], HARDWARE_KEYS),
            "classification": pick(timing["classification"], LATENCY_KEYS),
            "full_response": pick(timing["full_response"], LATENCY_KEYS),
            "artifact_total_bytes": timing["artifact_total_bytes"],
            "encoder_bytes": timing["encoder_bytes"],
            "measurement": "in_process_single_requests; no HTTP or concurrency; RSS is not peak",
        },
        "data.json": {
            "run_id": run_id,
            **pick(
                audit,
                (
                    "original_train",
                    "clean_train",
                    "excluded",
                    "exclusion_reasons",
                    "splits",
                    "empty_texts",
                    "missing_values",
                    "text_length_characters",
                    "test_unchanged",
                    "test_labels_used_for_cleanup",
                ),
            ),
            "official_test": test["n"],
            "classes": len(selection["labels"]),
            "seed": splits["seed"],
            "preprocessing": splits["preprocessing"],
            "split_fingerprints": {
                name: pick(row, ("rows", "assignment_hash", "class_counts"))
                for name, row in splits["splits"].items()
            },
            "dataset_commit": source["dataset_commit"],
            "encoder": pick(source["encoder"], ("id", "revision", "license")),
            "source_manifest_sha256": sha(artifact / "source.json"),
            "split_manifest_sha256": sha(artifact / "splits.json"),
        },
    }
    example = report / "http_example_response.json"
    if example.exists():
        public["example_response.json"] = {
            "request": {"text": "My card payment appears twice in the app."},
            "response": pick(
                read_json(example),
                (
                    "run_id",
                    "predicted_category",
                    "confidence",
                    "top_predictions",
                    "routing_decision",
                    "threshold",
                    "warnings",
                ),
            ),
            "scope": "shortened saved HTTP response; request from scripts/smoke_http.py",
        }
    output.mkdir(parents=True, exist_ok=True)
    for name, value in public.items():
        write_json(output / name, value)
    csv_names = ["validation_results.csv", "per_class.csv", "top_confusions.csv"]
    for name in csv_names:
        (output / name).write_bytes((report / name).read_text(encoding="utf-8-sig").encode("utf-8"))
    inputs = [
        artifact / n
        for n in (
            "selection.json",
            "policy.json",
            "manifest.json",
            "source.json",
            "splits.json",
        )
    ] + [
        report / n
        for n in (
            "test_metrics.json",
            "latency.json",
            "data_audit.json",
            "confusion_matrix.csv",
            "errors.csv",
            *csv_names,
        )
    ]
    if example.exists():
        inputs += [example, root / "scripts/smoke_http.py"]
    analysis = root / "reports/finishing/error_analysis.csv"
    completion = root / "reports/finishing/completion.json"
    include_analysis = (
        analysis.exists()
        and completion.exists()
        and run_id == read_json(completion)["original_run_id"]
    )
    if include_analysis:
        (output / "error_analysis.csv").write_bytes(
            analysis.read_text(encoding="utf-8-sig").encode("utf-8")
        )
        inputs.append(analysis)
    outputs = list(public) + csv_names
    if include_analysis:
        outputs.append("error_analysis.csv")
    write_json(
        output / "provenance.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "generator": "scripts/write_run_summary.py (scripts/public_reports.py)",
            "scope": "allowlisted projections; original local reports and artifacts unchanged",
            "sources": {
                p.relative_to(root).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
                for p in inputs
            },
            "outputs": {
                name: {"sha256": sha(output / name), "bytes": (output / name).stat().st_size}
                for name in outputs
            },
        },
    )
    rows = []
    for row in public["selection.json"]["candidates"]:
        m, t = row["metrics"], row["latency"]
        rows.append(
            f"| {row['id']} | {m['macro_f1']:.4f} | {m['accuracy']:.4f} | "
            f"{m['top3_accuracy']:.4f} | {row['training_seconds']:.2f} | "
            f"{t['median_ms']:.2f} / {t['p95_ms']:.2f} |"
        )
    auto, manual = counts["auto_route"], counts["manual_review"]
    text = f"""# Эксперименты

Run `{run_id}`. Сводка программно получена из сохранённых первичных отчётов:
[происхождение и SHA-256](../reports/public/provenance.json).
Генератор `scripts/write_run_summary.py` не обучает модели и не переписывает README.

## Шесть конфигураций — validation_model, n={selection["winner"]["metrics"]["n"]}

| Модель | Macro-F1 | Accuracy | Top-3 | Fit, с | Median / p95, мс |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

TF-IDF: word n-grams (1,2), min_df=2, max_features=50000, sublinear_tf=True.
MiniLM: frozen all-MiniLM-L6-v2, нормализованные float32 embeddings размерности 384,
максимум 256 токенов. LR: lbfgs, max_iter=2000, seed=42, C={{0.5,1,4}}.
Все кандидаты сошлись. Fit TF-IDF включает fit преобразователя и LR; fit MiniLM
здесь означает только LR на готовых embeddings. Вычисление embeddings в fit не входит.

Выбор до test: максимум macro-F1 по всем 77 классам; варианты в пределах 0.005
сравниваются по median latency. При разнице latency до 5% предпочтение TF-IDF,
затем меньшему C. Выбран `{selection["winner"]["id"]}`; tie-break не понадобился.
Baseline test отдельно не оценивался. [Полные числа](../reports/public/selection.json).

## Policy — отдельный validation_policy

Порог `{policy["threshold"]}` выбран после фиксации модели: максимальная coverage
при accepted accuracy ≥ {policy["target_accuracy"]:.0%} и accepted count ≥ {policy["min_accepted"]}.
На policy split приняты {policy["validation_metrics"]["accepted_count"]} из {policy["n"]},
правильных {policy["validation_metrics"]["accepted_correct"]}. После выбора refit нет.
Confidence = max predict_proba; threshold — сравниваемый с ним порог, не вероятность.
[Параметры policy и привязка к SHA-256 модели](../reports/public/policy.json).

## Official test — n={test["n"]}

Macro-F1 {test["macro_f1"]:.6f}; accuracy {test["accuracy"]:.6f};
top-3 {test["top3_accuracy"]:.6f} ({counts["top3_correct"]} правильных).

| Решение | Всего | Правильных top-1 | Ошибок top-1 |
| --- | ---: | ---: | ---: |
| auto_route | {auto["total"]} | {auto["correct"]} | {auto["errors"]} |
| manual_review | {manual["total"]} | {manual["correct"]} | {manual["errors"]} |
| all | {counts["all"]["total"]} | {counts["all"]["correct"]} | {counts["all"]["errors"]} |

Порог перехватывает {manual["errors"] / counts["all"]["errors"]:.2%} ошибок, но оставляет
{auto["errors"] / auto["total"]:.2%} ошибок среди автоматических решений. Top-3 не измеряет
точность человека. Confidence не калиброван; Wilson 95% для accepted accuracy
[{test["routing"]["accepted_accuracy_wilson95"][0]:.2%}, {test["routing"]["accepted_accuracy_wilson95"][1]:.2%}]
пересекает 95%, поэтому SLA не доказан.
[Test JSON](../reports/public/benchmark.json), [per-class](../reports/public/per_class.csv),
[путаницы](../reports/public/top_confusions.csv), [разбор ошибок](error_analysis.md).

## Производительность

{timing["hardware"]["cpu"]}, {timing["hardware"]["platform"]}, Python {timing["hardware"]["python"]},
{timing["threads"]} вычислительных потока, RAM {timing["hardware"]["ram_bytes"] / 2**30:.1f} GiB.
10 warmup + 100 фиксированных одиночных test-запросов, batch_size=1.
Классификация: median {timing["classification"]["median_ms"]:.2f} мс,
p95 {timing["classification"]["p95_ms"]:.2f} мс; с retrieval:
median {timing["full_response"]["median_ms"]:.2f} мс, p95 {timing["full_response"]["p95_ms"]:.2f} мс.
Это замер внутри процесса, без HTTP, сети и конкурентной нагрузки. Серии независимы:
немного меньший p95 полного ответа не означает ускорение за счёт retrieval.
RSS {timing["hardware"]["process_rss_bytes"] / 2**20:.1f} MiB — снимок, не peak.
HTTP latency и QPS не измерялись. [Условия, sample IDs и версии](../reports/public/latency.json).
"""
    (root / "docs/experiments.md").write_text(text, encoding="utf-8")
    return public


def export_reproduction(root=ROOT):
    root = Path(root)
    local, output = root / "reports/finishing", root / "reports/public"
    receipt = read_json(local / "clean_reproduction.json")
    completion = read_json(local / "completion.json")
    audit = read_json(local / "primary_report_audit.json")
    assert receipt["status"] == audit["status"] == "passed"
    linux = local / "linux-reproduction"
    policy = read_json(linux / "artifact_policy.json")
    stages = {}
    for command in receipt["commands"]:
        if command["exit_code"] == 0:
            stages[command["name"]] = pick(command, ("name", "exit_code", "seconds"))
    value = {
        **pick(
            receipt,
            (
                "status",
                "run_id",
                "reproduction_of",
                "platform",
                "test_previously_viewed",
                "purpose",
                "project_caches_before_start",
                "project_outputs_before_start",
                "manually_copied_resource_files",
                "original_experiment_unchanged",
                "frozen_artifact_hashes_unchanged_after_restart",
            ),
        ),
        "dependencies": "uv sync --frozen --no-dev --no-cache",
        "successful_stages": list(stages.values()),
        "test": pick(read_json(linux / "test_metrics.json"), TEST_KEYS),
        "policy": pick(policy, POLICY_KEYS),
        "full_ml_reproduction_count": completion["full_ml_reproduction_count"],
        "primary_audit": {"status": audit["status"], "checks": audit["checks"]},
        "original_hashes_unchanged": completion["original_hashes_unchanged"],
        "manual_review": {
            "status": completion["visual_browser_check"],
            "confirmed": read_json(local / "manual_ui_review.json")["confirmed"],
            "not_confirmed": read_json(local / "manual_ui_review.json")["not_confirmed"],
            "screenshots_in_repository": False,
            "student_error_analysis_performed": False,
        },
        "provenance": {
            p.relative_to(root).as_posix(): {"sha256": sha(p)}
            for p in [
                local / "clean_reproduction.json",
                local / "completion.json",
                local / "primary_report_audit.json",
                local / "manual_ui_review.json",
                linux / "test_metrics.json",
                linux / "artifact_policy.json",
            ]
        },
    }
    write_json(output / "reproducibility.json", value)
    return value


def main(kind):
    parser = argparse.ArgumentParser(description=__doc__)
    if kind == "run":
        parser.add_argument("--run-id", help="Completed local run; defaults to active run")
    args = parser.parse_args()
    if kind == "run":
        export_run(run_id=args.run_id)
    else:
        export_reproduction()
    print("Public evidence updated from saved reports; README and original artifacts unchanged")
