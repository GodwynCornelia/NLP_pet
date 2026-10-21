import time
from pathlib import Path
from uuid import uuid4

import joblib

from support_router.common import (
    PREPROCESSING,
    check_file,
    digest,
    read_json,
    run_paths,
    sha,
    versions,
)
from support_router.data import normalize, verify_source
from support_router.models import encode, load_encoder, stable_top
from support_router.policy import route
from support_router.retrieval import RetrievalIndex


def verify_manifest(config, run_id):
    run_id, artifact, _ = run_paths(config, run_id)
    manifest = read_json(artifact / "manifest.json")
    if manifest["schema_version"] != 1 or manifest["run_id"] != run_id:
        raise ValueError("unsupported_or_mismatched_manifest")
    required = {
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
        manifest["classifier"]["file"],
    }
    if not required.issubset(manifest["files"]):
        raise ValueError("required_artifact_missing_from_manifest")
    for relative, info in manifest["files"].items():
        if Path(relative).name != relative:
            raise ValueError("invalid_artifact_relative_path")
        check_file(artifact / relative, info)
    for key, file in [
        ("source_manifest_hash", "source.json"),
        ("split_manifest_hash", "splits.json"),
        ("policy_hash", "policy.json"),
        ("index_hash", "index.json"),
    ]:
        if manifest[key] != sha(artifact / file):
            raise ValueError(f"manifest_reference_mismatch: {key}")
    installed = versions()
    for package in ["scikit-learn", "numpy", "joblib", "sentence-transformers", "torch"]:
        if installed[package] != manifest["versions"][package]:
            raise ValueError(f"incompatible_library_version: {package}")
    if manifest["preprocessing"] != PREPROCESSING:
        raise ValueError("preprocessing_version_mismatch")
    source = read_json(artifact / "source.json")
    verify_source(Path(config["project"]["data_dir"]), source)
    selection = read_json(artifact / "selection.json")
    policy = read_json(artifact / "policy.json")
    index = read_json(artifact / "index.json")
    run = read_json(artifact / "run.json")
    if run["run_id"] != run_id or run["config_hash"] != manifest["config_hash"]:
        raise ValueError("run_config_mismatch")
    if digest(read_json(artifact / "config.json")) != run["config_hash"]:
        raise ValueError("saved_config_hash_mismatch")
    if selection["source_manifest_hash"] != sha(artifact / "source.json"):
        raise ValueError("selection_source_mismatch")
    check_file(artifact / selection["winner"]["file"], selection["winner"]["artifact"])
    for component in [selection, policy, index]:
        if component["run_id"] != run_id:
            raise ValueError("component_run_id_mismatch")
    if policy["classifier_sha256"] != selection["winner"]["artifact"]["sha256"]:
        raise ValueError("policy_classifier_mismatch")
    if policy["selection_sha256"] != sha(artifact / "selection.json"):
        raise ValueError("policy_selection_mismatch")
    split_hash = sha(artifact / "splits.json")
    if any(
        component.get(key) != split_hash
        for component, key in [
            (selection, "split_manifest_hash"),
            (policy, "split_manifest_sha256"),
            (index, "split_manifest_sha256"),
        ]
    ):
        raise ValueError("component_split_mismatch")
    if manifest["classifier"] != selection["winner"]:
        raise ValueError("classifier_selection_mismatch")
    if not policy["auto_routing_enabled"] and policy["threshold"] is not None:
        raise ValueError("disabled_policy_must_have_null_threshold")
    if policy["auto_routing_enabled"] and (
        policy["threshold"] is None or not 0 <= policy["threshold"] <= 1
    ):
        raise ValueError("invalid_policy_threshold")
    if index["encoder_revision"] != source["encoder"]["revision"]:
        raise ValueError("index_encoder_mismatch")
    return manifest


class Router:
    def __init__(self, config, run_id=None):
        self.config = config
        self.run_id, self.artifact, self.report = run_paths(config, run_id)
        self.manifest = verify_manifest(config, self.run_id)
        self.selection = read_json(self.artifact / "selection.json")
        self.policy = read_json(self.artifact / "policy.json")
        winner = self.selection["winner"]
        self.family = winner["family"]
        # Only this trusted local pipeline writes joblib. No upload/deserialization endpoint exists.
        self.model = joblib.load(self.artifact / winner["file"])
        self.classes = self.model.classes_.tolist()
        if (
            self.classes != self.manifest["labels"]
            or self.classes != self.manifest["classifier_classes"]
        ):
            raise ValueError("classifier_label_order_mismatch")
        if len(self.classes) != 77:
            raise ValueError("expected_77_classes")
        self.encoder = load_encoder(config, read_json(self.artifact / "source.json"))
        metadata = self.selection["encoder"]
        if (
            self.encoder.max_seq_length != metadata["max_seq_length"]
            or (
                self.encoder.get_embedding_dimension()
                if hasattr(self.encoder, "get_embedding_dimension")
                else self.encoder.get_sentence_embedding_dimension()
            )
            != metadata["dimension"]
        ):
            raise ValueError("encoder_dimension_or_token_limit_mismatch")
        expected_ids = read_json(self.artifact / "splits.json")["splits"]["train_fit"]["sample_ids"]
        if digest(expected_ids) != self.manifest["train_fit_sample_ids_hash"]:
            raise ValueError("train_fit_ids_mismatch")
        self.index = RetrievalIndex(
            self.artifact, read_json(self.artifact / "index.json"), expected_ids
        )
        if self.index.matrix.shape[1] != metadata["dimension"]:
            raise ValueError("index_encoder_dimension_mismatch")
        if self.family == "minilm" and self.model.n_features_in_ != metadata["dimension"]:
            raise ValueError("classifier_encoder_dimension_mismatch")

    def predict_many(self, texts, top_k=3, include_similar=True):
        started = time.perf_counter()
        normalized = [normalize(t) for t in texts]
        need_embeddings = self.family == "minilm" or include_similar
        embeddings = (
            encode(self.encoder, normalized, self.config["model"]["batch_size"])
            if need_embeddings
            else None
        )
        inputs = normalized if self.family == "tfidf" else embeddings
        probabilities = self.model.predict_proba(inputs)
        lengths = None
        if need_embeddings:
            tokens = self.encoder.tokenizer(
                normalized, truncation=False, padding=False, return_length=True, verbose=False
            )
            lengths = tokens["length"]
        results = []
        for i, probs in enumerate(probabilities):
            top = stable_top(probs, self.classes, top_k)
            decision, reason = route(top[0]["score"], self.policy)
            similar = self.index.search(embeddings[i], texts[i]) if include_similar else []
            results.append(
                {
                    "request_id": str(uuid4()),
                    "run_id": self.run_id,
                    "predicted_category": top[0]["category"],
                    "confidence": top[0]["score"],
                    "top_predictions": top,
                    "routing_decision": decision,
                    "review_reason": reason,
                    "threshold": self.policy["threshold"],
                    "similar_examples": similar,
                    "warnings": ["embedding_input_truncated"]
                    if lengths is not None and lengths[i] > self.encoder.max_seq_length
                    else [],
                    "latency_ms": 0.0,
                }
            )
        elapsed = (time.perf_counter() - started) * 1000
        for result in results:
            result["latency_ms"] = elapsed  # Total synchronous processing time for this batch.
        return results

    def predict(self, text, top_k=3, include_similar=True):
        return self.predict_many([text], top_k, include_similar)[0]

    def metadata(self):
        return {
            "api_version": "0.1.0",
            "run_id": self.run_id,
            "model": self.selection["winner"]["id"],
            "supported_language": "en",
            "categories": self.classes,
            "threshold": self.policy["threshold"],
            "auto_routing_enabled": self.policy["auto_routing_enabled"],
            "limits": self.config["limits"],
            "embedding_max_tokens": self.encoder.max_seq_length,
            "truncation": "Embedding inputs beyond max tokens are truncated",
            "limitations": [
                "English banking queries only",
                "Not an OOD detector",
                "Confidence is uncalibrated",
                "Similarity is not an explanation",
            ],
        }
