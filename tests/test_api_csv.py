import logging

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from support_router.api.main import create_app
from support_router.config import load_config
from support_router.csv_io import csv_batches, export_csv, read_upload, results_frame


class FakeRouter:
    """Contract fixture only; never used in benchmark reports."""

    run_id = "unit-fixture"

    def predict(self, text, top_k=3, include_similar=True):
        return {
            "request_id": "fixture",
            "run_id": self.run_id,
            "predicted_category": "a",
            "confidence": 0.6,
            "top_predictions": [{"category": "a", "score": 0.6}],
            "routing_decision": "manual_review",
            "review_reason": "low_confidence",
            "threshold": 0.8,
            "similar_examples": [],
            "warnings": [],
            "latency_ms": 1.0,
        }

    def predict_many(self, texts, top_k, include_similar):
        return [self.predict(t, top_k, include_similar) for t in texts]


@pytest.fixture
def client():
    with TestClient(create_app(router_factory=FakeRouter)) as client:
        yield client


@pytest.mark.parametrize(
    "body",
    [
        {"text": " "},
        {"text": "x" * 4001},
        {"text": 4},
        {"text": "a", "top_k": 0},
        {"text": "a", "top_k": 6},
        {"text": "a", "top_k": True},
    ],
)
def test_invalid_single(client, body):
    assert client.post("/predict", json=body).status_code == 422


def test_missing_artifacts_health_ready_503(tmp_path):
    config = load_config()
    config["project"]["artifact_dir"] = str(tmp_path)
    with TestClient(create_app(config)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json={"text": "hello"}).status_code == 503


def test_batch_preserves_ids_order_and_row_errors(client):
    response = client.post(
        "/predict-batch",
        json={
            "items": [
                {"id": "b", "text": "valid text"},
                {"id": "a", "text": ""},
                {"id": "c", "text": {"invalid": "object"}},
            ]
        },
    )
    assert response.status_code == 200
    rows = response.json()
    assert [r["id"] for r in rows] == ["b", "a", "c"]
    assert rows[0]["result"] and rows[0]["error"] is None
    assert rows[1]["result"] is None and rows[1]["error"]
    assert rows[2]["result"] is None and rows[2]["error"]
    assert (
        client.post(
            "/predict-batch",
            json={"items": [{"id": "a", "text": "hello"}, {"id": "a", "text": "world"}]},
        ).status_code
        == 422
    )
    assert client.post("/predict-batch", json={"items": []}).status_code == 422
    assert (
        client.post(
            "/predict-batch", json={"items": [{"id": str(i), "text": "hello"} for i in range(101)]}
        ).status_code
        == 422
    )


def test_body_limit_and_no_raw_text_in_logs(client, caplog):
    with caplog.at_level(logging.INFO):
        response = client.post("/predict", json={"text": "PRIVATE_BANK_QUERY_123"})
    assert response.status_code == 200
    assert "PRIVATE_BANK_QUERY_123" not in caplog.text
    assert response.headers["x-request-id"] == response.json()["request_id"]
    assert client.post("/predict", content=b"x" * 2097153).status_code == 413
    assert client.post("/predict", content=iter([b"x" * 1048576] * 3)).status_code == 413


def test_unexpected_error_has_request_id_without_trace():
    class Broken(FakeRouter):
        def predict(self, *args):
            raise RuntimeError("private text should never be returned")

    with TestClient(create_app(router_factory=Broken), raise_server_exceptions=False) as client:
        response = client.post("/predict", json={"text": "test"})
        assert response.status_code == 500
        assert response.json()["request_id"]
        assert "private text" not in response.text


def test_csv_validation_preserves_na_invalid_rows_and_export(client):
    limits = load_config()["limits"]
    frame = read_upload(b"id,text,extra\na,NA,=2+3\na,,safe\nb,hello,@evil\n", limits)
    assert frame.text.tolist() == ["NA", "", "hello"]
    items = list(csv_batches(frame))[0]
    rows = client.post("/predict-batch", json={"items": items}).json()
    output = results_frame(frame, rows)
    assert len(output) == len(frame)
    assert output.loc[1, "error"]
    assert output.id.tolist() == ["a", "a", "b"]
    exported = export_csv(output).decode("utf-8-sig")
    assert "'=2+3" in exported and "'@evil" in exported
    assert frame.loc[0, "extra"] == "=2+3"
    for content in [
        b"wrong\nhello",
        b"text\n\xff",
        b"",
        b"text\n" + b"a\n" * 1001,
        b"x" * (limits["csv_bytes"] + 1),
    ]:
        with pytest.raises(ValueError):
            read_upload(content, limits)


def test_formula_safe_export_handles_prefix_whitespace():
    data = pd.DataFrame({"text": ["=1", " +1", "-3", "@x", "\t=1", "\r=2", "safe"]})
    exported = export_csv(data).decode("utf-8-sig")
    reread = pd.read_csv(__import__("io").StringIO(exported), keep_default_na=False)
    assert all(x.startswith("'") for x in reread.text.iloc[:6])
    assert reread.text.iloc[-1] == "safe"


def test_csv_completely_empty_record_is_retained_as_invalid_item(client):
    frame = read_upload(b"text\nfirst\n\nlast\n", load_config()["limits"])
    assert frame.text.tolist() == ["first", "", "last"]
    rows = client.post("/predict-batch", json={"items": list(csv_batches(frame))[0]}).json()
    assert len(results_frame(frame, rows)) == 3
    assert rows[1]["error"] and rows[1]["result"] is None
