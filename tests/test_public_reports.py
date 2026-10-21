"""Check published evidence and prevent accidental export of private metadata."""

import copy
import json
import runpy

import numpy as np
import pandas as pd
import pytest

from support_router.common import read_json, sha
from support_router.config import ROOT

EXPORT = runpy.run_path(str(ROOT / "scripts/public_reports.py"))


def test_decision_counts_reject_inconsistent_routing():
    matrix = np.array([[1, 1], [0, 1]])
    errors = pd.DataFrame(
        [{"true_category": "b", "top3": '[{"category":"b"}]', "routing_decision": "manual_review"}]
    )
    metrics = {
        "n": 3,
        "accuracy": 2 / 3,
        "top3_accuracy": 1.0,
        "routing": {"accepted_count": 1, "accepted_correct": 1},
    }
    result = EXPORT["decision_counts"](metrics, matrix, errors)
    assert result["manual_review"] == {"total": 2, "correct": 1, "errors": 1}
    assert result["top3_correct"] == 3
    metrics["routing"]["accepted_count"] = 2
    with pytest.raises(ValueError, match="routing_error_count_mismatch"):
        EXPORT["decision_counts"](metrics, matrix, errors)


def test_selection_export_excludes_private_tokenizer_metadata():
    source = copy.deepcopy(read_json(ROOT / "reports/public/selection.json"))
    source["winner"] = {"id": source["winner"]}
    source["encoder"] = {"tokenizer_parameters": "private-home/vocab.txt"}
    source["hardware"] = {"node": "private-hostname"}
    source["local_credentials"] = "private-token-value"
    public = EXPORT["public_selection"](source)
    assert "private-" not in json.dumps(public)
    assert public["winner"] == "minilm-C4"
    assert len(public["candidates"]) == 6


def test_published_evidence_has_consistent_provenance_and_counts():
    directory = ROOT / "reports/public"
    provenance = read_json(directory / "provenance.json")
    for name, metadata in provenance["outputs"].items():
        assert sha(directory / name) == metadata["sha256"], name
        assert b"\r\n" not in (directory / name).read_bytes(), name
    benchmark = read_json(directory / "benchmark.json")
    counts = benchmark["decision_counts"]
    for key in ("total", "correct", "errors"):
        assert counts["auto_route"][key] + counts["manual_review"][key] == counts["all"][key]
    assert counts["all"]["total"] == benchmark["n"]
    assert counts["all"]["correct"] / benchmark["n"] == benchmark["accuracy"]
    assert counts["top3_correct"] / benchmark["n"] == benchmark["top3_accuracy"]
    assert benchmark["run_id"] == provenance["run_id"]
