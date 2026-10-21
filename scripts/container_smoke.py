"""HTTP/CSV checks from the UI container with external sockets forbidden."""

import json
import os
import socket
from urllib.parse import urlparse

import httpx

from support_router.config import load_config
from support_router.csv_io import csv_batches, export_csv, read_upload, results_frame


def main():
    api = os.getenv("API_BASE_URL", "http://api:8000")
    ui = os.getenv("UI_BASE_URL", "http://ui:8501")
    allowed = {"127.0.0.1", "::1"}
    for base in [api, ui]:
        host = urlparse(base).hostname
        allowed.add(host)
        allowed.update(info[4][0] for info in socket.getaddrinfo(host, None))
    original_connect, original_create = socket.socket.connect, socket.create_connection

    def check(address):
        if not isinstance(address, tuple) or address[0] not in allowed:
            raise RuntimeError("external_socket_forbidden")

    def connect(sock, address):
        check(address)
        return original_connect(sock, address)

    def create(address, *args, **kwargs):
        check(address)
        return original_create(address, *args, **kwargs)

    socket.socket.connect, socket.create_connection = connect, create
    try:
        socket.create_connection(("huggingface.co", 443))
        raise AssertionError("External connection was allowed")
    except RuntimeError as exc:
        assert str(exc) == "external_socket_forbidden"
    with httpx.Client(base_url=api, timeout=120, trust_env=False) as client:
        assert client.get("/health").status_code == 200
        ready = client.get("/ready")
        ready.raise_for_status()
        run_id = ready.json()["run_id"]
        meta = client.get("/metadata").json()
        assert meta["run_id"] == run_id and len(meta["categories"]) == 77
        result = client.post("/predict", json={"text": "My card payment appears twice in the app."})
        result.raise_for_status()
        prediction = result.json()
        assert prediction["predicted_category"] == "transaction_charged_twice"
        assert len(prediction["top_predictions"]) == len(prediction["similar_examples"]) == 3
        assert client.post("/predict", json={"text": " "}).status_code == 422
        long = client.post("/predict", json={"text": "bank " * 780})
        assert long.status_code == 200 and "embedding_input_truncated" in long.json()["warnings"]
        frame = read_upload(
            b"id,text,extra\na,Where is my card?,=2+3\na,,safe\nb,NA,keep\n",
            load_config()["limits"],
        )
        results = []
        for items in csv_batches(frame):
            response = client.post("/predict-batch", json={"items": items})
            response.raise_for_status()
            results.extend(response.json())
        output = results_frame(frame, results)
        assert output.id.tolist() == ["a", "a", "b"]
        assert output.text.tolist() == frame.text.tolist() and len(output) == 3
        assert results[1]["error"] and results[1]["result"] is None
        assert "'=2+3" in export_csv(output).decode("utf-8-sig")
        reports = client.get("/reports")
        reports.raise_for_status()
        assert reports.json()["run_id"] == run_id
        assert reports.json()["test"]["run_id"] == run_id
        test_metrics = reports.json()["test"]
    assert httpx.get(ui + "/_stcore/health", timeout=15, trust_env=False).status_code == 200
    assert httpx.get(ui, timeout=15, trust_env=False).status_code == 200
    from streamlit.testing.v1 import AppTest

    from support_router.config import ROOT

    app = AppTest.from_file(str(ROOT / "src/support_router/ui/app.py"), default_timeout=120).run()
    assert not app.exception, [e.message for e in app.exception]
    assert [tab.label for tab in app.tabs] == [
        "Проверить обращение",
        "Обработать CSV",
        "Качество модели",
    ]
    assert app.metric[0].value == f"{test_metrics['macro_f1']:.4f}"
    app.text_area[0].input("My card payment appears twice in the app.").run()
    app.button[0].click().run()
    assert not app.exception, [e.message for e in app.exception]
    assert any(e.value == "transaction_charged_twice" for e in app.subheader)
    assert any(e.value == "Похожие обращения из train_fit" for e in app.subheader)
    print(
        json.dumps(
            {
                "mode": "smoke",
                "status": "passed",
                "run_id": run_id,
                "external_sockets": "forbidden",
                "internal_ui_api_http": "passed",
                "streamlit_apptest": "passed; actual UI script fetched the API and predicted",
                "checks": [
                    "health",
                    "ready",
                    "metadata",
                    "predict",
                    "batch_order_and_error",
                    "csv_export",
                    "empty_422",
                    "truncation",
                    "report_run",
                    "ui_http",
                ],
                "browser_visual_check": False,
            }
        )
    )


if __name__ == "__main__":
    main()
