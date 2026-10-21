import inspect
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
import pytest

from support_router.common import check_file, file_info
from support_router.config import load_config
from support_router.data import clean_training, duplicate_key, load_split, normalize, split_training
from support_router.models import baseline, embedding_key, save_model, stable_top
from support_router.policy import choose_policy, policy_stats, route
from support_router.training import select_candidate, train, tune_policy


def test_normalization_preserves_meaning():
    assert normalize("  I do NOT pay 42!\nCafe\u0301\t ") == "I do NOT pay 42! Café"
    assert duplicate_key(" A\tCARD ") == duplicate_key("a card")


def test_cleanup_and_split_exclude_conflicts_duplicates_and_test():
    rows = [
        {"sample_id": f"{c}-{i}", "text": f"{c} sample {i}", "category": c}
        for c in ["a", "b", "c"]
        for i in range(30)
    ]
    rows += [
        dict(rows[0], sample_id="duplicate"),
        {"sample_id": "conflict1", "text": "conflict", "category": "a"},
        {"sample_id": "conflict2", "text": " CONFLICT ", "category": "b"},
    ]
    raw = pd.DataFrame(rows)
    original = raw.copy(deep=True)
    clean, excluded = clean_training(raw, ["a sample 1"])
    pd.testing.assert_frame_equal(raw, original)
    assert excluded.reason.value_counts().to_dict() == {
        "conflicting_labels": 2,
        "official_test_overlap": 1,
        "duplicate_same_label": 1,
    }
    splits = split_training(clean, expected_classes=3)
    repeated = split_training(clean, expected_classes=3)
    groups, ids = set(), set()
    for name, frame in splits.items():
        assert not groups.intersection(frame.group_id)
        assert not ids.intersection(frame.sample_id)
        groups.update(frame.group_id)
        ids.update(frame.sample_id)
        pd.testing.assert_frame_equal(frame, repeated[name])
        assert frame.category.nunique() == 3


def test_training_loader_cannot_load_test_and_training_has_no_test_access():
    with pytest.raises(ValueError, match="cannot load"):
        load_split({}, "unused", "test")
    for function in [train, tune_policy]:
        assert '"test"' not in inspect.getsource(function)
    assert '"validation_policy"' in inspect.getsource(tune_policy)


def test_tfidf_fit_only_and_serialization_and_class_order(tmp_path):
    model = baseline(1, load_config())
    model.fit(
        ["card broken", "card broken", "transfer money", "transfer money"], ["z", "z", "a", "a"]
    )
    model.predict_proba(["validationonlyword"])
    assert "validationonlyword" not in model["tfidf"].vocabulary_
    path = tmp_path / "trusted.joblib"
    save_model(path, model)
    restored = joblib.load(path)
    probs = model.predict_proba(["card broken"])[0]
    np.testing.assert_allclose(restored.predict_proba(["card broken"])[0], probs)
    assert stable_top(probs, model.classes_, 2)[0]["category"] == "z"
    assert stable_top(np.array([0.5, 0.5]), ["z", "a"])[0]["category"] == "z"


@pytest.mark.parametrize("change", ["revision", "text", "order", "preprocessing", "tokenization"])
def test_embedding_cache_invalidates(change):
    frame = pd.DataFrame({"sample_id": ["1", "2"], "normalized": ["A", "B"]})
    meta = {"revision": "old", "preprocessing": "nfc", "tokenization": {"max": 256}}
    original = embedding_key(frame, meta)
    if change == "text":
        frame.loc[0, "normalized"] = "C"
    elif change == "order":
        frame = frame.iloc[::-1]
    else:
        meta[change] = "changed"
    assert embedding_key(frame, meta) != original


def test_hash_detects_changed_file_without_overwriting(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_bytes(b"original")
    info = file_info(path)
    check_file(path, info)
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash_mismatch"):
        check_file(path, info)
    assert path.read_bytes() == b"modified"


def test_policy_target_boundary_ties_and_zero_accepted():
    probs = np.array([[0.9, 0.1], [0.7, 0.3], [0.7, 0.3], [0.6, 0.4]])
    policy, rows = choose_policy(probs, ["a", "a", "a", "b"], ["a", "b"], 0.95, 2)
    assert policy["threshold"] == 0.7
    assert route(0.7, policy) == ("auto_route", None)
    assert route(0.699, policy) == ("manual_review", "low_confidence")
    assert len(rows) == 3
    disabled, _ = choose_policy(probs, ["b"] * 4, ["a", "b"], 0.95, 2)
    assert disabled["auto_routing_enabled"] is False
    assert disabled["threshold"] is None
    stats = policy_stats(probs, ["b"] * 4, ["a", "b"], disabled)
    assert stats["accepted_accuracy"] is None
    assert stats["accepted_accuracy_wilson95"] is None
    assert stats["accepted_count"] == 0
    assert route(1, disabled) == ("manual_review", "auto_routing_disabled")


def test_selection_f1_latency_rule():
    config = load_config()
    rows = [
        {"family": "minilm", "C": 4, "metrics": {"macro_f1": 0.85}, "latency": {"median_ms": 10}},
        {"family": "tfidf", "C": 1, "metrics": {"macro_f1": 0.847}, "latency": {"median_ms": 1}},
    ]
    assert select_candidate(rows, config)["family"] == "tfidf"
    changed = deepcopy(rows)
    changed[1]["metrics"]["macro_f1"] = 0.84
    assert select_candidate(changed, config)["family"] == "minilm"
