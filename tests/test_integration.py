import socket

import numpy as np
import pytest
from fastapi.testclient import TestClient

from support_router.api.main import create_app
from support_router.common import read_json, run_paths, setup_compute
from support_router.config import load_config
from support_router.data import load_split
from support_router.models import predict_proba
from support_router.runtime import Router

pytestmark = pytest.mark.integration


def test_real_artifacts_offline_predict_batch_ready_and_reports(monkeypatch):
    config = load_config()
    _controller = setup_compute(config)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    original_connect = socket.socket.connect
    original_create = socket.create_connection

    def local_only(address):
        # Windows asyncio itself needs a loopback socketpair. Keep loopback available
        # for UI/API as required, while rejecting every external hostname/address.
        if not isinstance(address, tuple) or address[0] not in {"127.0.0.1", "::1", "localhost"}:
            raise AssertionError("External network is forbidden during offline integration")

    def guarded_connect(sock, address):
        local_only(address)
        return original_connect(sock, address)

    def guarded_create(address, *args, **kwargs):
        local_only(address)
        return original_create(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", guarded_create)
    with pytest.raises(AssertionError, match="External network"):
        socket.create_connection(("huggingface.co", 443))
    router = Router(config)
    expected = router.predict("My card payment appears twice in the app.")
    probabilities = predict_proba(
        router.model, router.family, ["My card payment appears twice in the app."], router.encoder
    )[0]
    assert np.isfinite(probabilities).all() and abs(probabilities.sum() - 1) < 1e-5
    assert expected["predicted_category"] == router.classes[int(probabilities.argmax())]
    with TestClient(create_app(config, router_factory=lambda: router)) as client:
        assert client.get("/ready").json()["run_id"] == router.run_id
        assert client.get("/health").status_code == 200
        actual = client.post(
            "/predict", json={"text": "My card payment appears twice in the app."}
        ).json()
        for key in [
            "predicted_category",
            "top_predictions",
            "routing_decision",
            "similar_examples",
        ]:
            assert actual[key] == expected[key]
        response = client.post(
            "/predict-batch",
            json={
                "items": [
                    {"id": "first", "text": "My card payment appears twice in the app."},
                    {"id": "invalid", "text": " "},
                    {"id": "last", "text": "Where is my card?"},
                ]
            },
        )
        assert response.status_code == 200
        rows = response.json()
        assert [r["id"] for r in rows] == ["first", "invalid", "last"]
        assert rows[1]["error"] and rows[1]["result"] is None
        assert rows[0]["result"]["predicted_category"] == actual["predicted_category"]
        long = client.post("/predict", json={"text": "bank " * 780}).json()
        assert "embedding_input_truncated" in long["warnings"]
        report = client.get("/reports")
        assert report.status_code == 200 and report.json()["run_id"] == router.run_id
    _, artifact, _ = run_paths(config, router.run_id)
    fit = load_split(config, router.run_id, "train_fit")
    assert set(router.index.rows.sample_id) == set(fit.sample_id)
    manifest = read_json(artifact / "manifest.json")
    for split in ["validation_model", "validation_policy"]:
        held = load_split(config, router.run_id, split)
        assert not set(held.group_id) & set(fit.group_id)
    assert manifest["classifier_classes"] == router.classes
