from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from support_router.common import digest, file_info, sha, versions, write_json
from support_router.config import load_config
from support_router.retrieval import RetrievalIndex
from support_router.runtime import verify_manifest


@pytest.fixture
def bundle(tmp_path):
    """Small manifest fixture without external resources or real benchmark claims."""
    config = load_config()
    config["project"]["artifact_dir"] = str(tmp_path / "artifacts")
    config["project"]["data_dir"] = str(tmp_path / "data")
    artifact = tmp_path / "artifacts" / "fixture"
    artifact.mkdir(parents=True)
    (artifact / "model.joblib").write_bytes(b"not deserialized by manifest-only checks")
    write_json(artifact / "source.json", {"files": {}, "encoder": {"revision": "r1"}})
    write_json(artifact / "splits.json", {"splits": {"train_fit": {"sample_ids": ["fit-1"]}}})
    split_hash = sha(artifact / "splits.json")
    winner = {"file": "model.joblib", "artifact": file_info(artifact / "model.joblib")}
    selection = {
        "run_id": "fixture",
        "winner": winner,
        "split_manifest_hash": split_hash,
        "source_manifest_hash": sha(artifact / "source.json"),
    }
    write_json(artifact / "selection.json", selection)
    policy = {
        "run_id": "fixture",
        "classifier_sha256": winner["artifact"]["sha256"],
        "selection_sha256": sha(artifact / "selection.json"),
        "split_manifest_sha256": split_hash,
        "auto_routing_enabled": False,
        "threshold": None,
    }
    write_json(artifact / "policy.json", policy)
    index = {"run_id": "fixture", "encoder_revision": "r1", "split_manifest_sha256": split_hash}
    write_json(artifact / "index.json", index)
    write_json(artifact / "config.json", {})
    write_json(artifact / "run.json", {"run_id": "fixture", "config_hash": digest({})})
    for name in ["encoder.json", "index.csv", "index.npy"]:
        (artifact / name).write_bytes(b"manifest-only placeholder")
    files = [
        "run.json",
        "config.json",
        "encoder.json",
        "index.csv",
        "index.npy",
        "source.json",
        "splits.json",
        "selection.json",
        "policy.json",
        "index.json",
        "model.joblib",
    ]
    manifest = {
        "schema_version": 1,
        "run_id": "fixture",
        "preprocessing": "nfc-strip-whitespace-v1",
        "versions": versions(),
        "source_manifest_hash": sha(artifact / "source.json"),
        "split_manifest_hash": split_hash,
        "policy_hash": sha(artifact / "policy.json"),
        "index_hash": sha(artifact / "index.json"),
        "classifier": winner,
        "config_hash": digest({}),
        "files": {file: file_info(artifact / file) for file in files},
    }
    write_json(artifact / "manifest.json", manifest)
    return config, artifact, manifest, policy, index


def test_manifest_detects_tampering_before_deserialization(bundle):
    config, artifact, _, _, _ = bundle
    verify_manifest(config, "fixture")
    (artifact / "model.joblib").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash_mismatch"):
        verify_manifest(config, "fixture")


@pytest.mark.parametrize("kind", ["policy", "index", "run_id", "version"])
def test_semantic_mismatches_rejected_even_with_updated_file_checksums(bundle, kind):
    config, artifact, manifest, policy, index = bundle
    if kind in {"policy", "run_id"}:
        policy["classifier_sha256" if kind == "policy" else "run_id"] = "different"
        write_json(artifact / "policy.json", policy)
        manifest["policy_hash"] = sha(artifact / "policy.json")
        manifest["files"]["policy.json"] = file_info(artifact / "policy.json")
    elif kind == "index":
        index["encoder_revision"] = "different"
        write_json(artifact / "index.json", index)
        manifest["index_hash"] = sha(artifact / "index.json")
        manifest["files"]["index.json"] = file_info(artifact / "index.json")
    else:
        manifest["versions"]["scikit-learn"] = "0.0"
    write_json(artifact / "manifest.json", manifest)
    with pytest.raises(ValueError, match="mismatch|incompatible"):
        verify_manifest(config, "fixture")


def test_retrieval_fit_only_ids_dimensions_and_exact_match(tmp_path):
    frame = pd.DataFrame(
        {
            "sample_id": ["fit-1", "fit-2"],
            "text": ["My CARD", "Other"],
            "category": ["a", "b"],
            "group_id": ["g1", "g2"],
        }
    )
    frame.to_csv(tmp_path / "index.csv", index=False)
    np.save(tmp_path / "index.npy", np.eye(2, dtype=np.float32))
    metadata = {
        "source_split": "train_fit",
        "sample_ids": ["fit-1", "fit-2"],
        "dimension": 2,
        "files": {f: file_info(tmp_path / f) for f in ["index.csv", "index.npy"]},
    }
    index = RetrievalIndex(tmp_path, metadata, ["fit-1", "fit-2"])
    found = index.search(np.array([1, 0], dtype=np.float32), " my card ")
    assert found[0]["sample_id"] == "fit-1" and found[0]["exact_match"]
    assert found[0]["similarity"] == 1
    with pytest.raises(ValueError, match="split_mismatch"):
        RetrievalIndex(tmp_path, metadata, ["test-1", "fit-2"])
    changed = deepcopy(metadata)
    changed["dimension"] = 3
    with pytest.raises(ValueError, match="shape_mismatch"):
        RetrievalIndex(tmp_path, changed, metadata["sample_ids"])
