"""Explicit HTTP/CSV smoke against a running local API and Streamlit process."""

import os

import httpx

from support_router.common import now, run_paths, write_json
from support_router.config import load_config
from support_router.csv_io import csv_batches, export_csv, read_upload, results_frame


def main():
    config = load_config()
    run_id, _, report = run_paths(config)
    base = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    with httpx.Client(base_url=base, timeout=120) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").json()["run_id"] == run_id
        assert len(client.get("/metadata").json()["categories"]) == 77
        response = client.post(
            "/predict", json={"text": "My card payment appears twice in the app."}
        )
        response.raise_for_status()
        prediction = response.json()
        write_json(report / "http_example_response.json", prediction)
        assert prediction["request_id"] == response.headers["x-request-id"]
        assert len(prediction["similar_examples"]) == 3
        assert client.post("/predict", json={"text": " "}).status_code == 422
        assert client.post("/predict", content=b"x" * 2097153).status_code == 413
        frame = read_upload(
            b"id,text,extra\na,NA,=2+3\na,,keep\nb,Where is my card?,safe\n", config["limits"]
        )
        results = []
        for items in csv_batches(frame):
            result = client.post("/predict-batch", json={"items": items})
            result.raise_for_status()
            results.extend(result.json())
        output = results_frame(frame, results)
        assert len(output) == 3 and output.text.tolist() == frame.text.tolist()
        assert results[1]["result"] is None and results[1]["error"]
        assert "'=2+3" in export_csv(output).decode("utf-8-sig")
        reports = client.get("/reports")
        assert reports.status_code == 200 and reports.json()["run_id"] == run_id
        health = httpx.get("http://127.0.0.1:8501/_stcore/health", timeout=10)
        assert health.status_code == 200
    write_json(
        report / "http_smoke.json",
        {
            "run_id": run_id,
            "mode": "smoke",
            "created_at": now(),
            "status": "passed",
            "checks": [
                "health",
                "ready",
                "metadata",
                "real_predict",
                "batch_item_error",
                "csv_order_and_safe_export",
                "422",
                "413",
                "reports",
                "streamlit_health",
            ],
        },
    )
    print(f"HTTP/CSV smoke passed: {run_id}")


if __name__ == "__main__":
    main()
