"""Render Streamlit and interact with real API; requires API_BASE_URL or localhost:8000."""

from streamlit.testing.v1 import AppTest

from support_router.common import now, run_paths, write_json
from support_router.config import ROOT, load_config


def main():
    at = AppTest.from_file(str(ROOT / "src/support_router/ui/app.py"), default_timeout=120).run()
    assert not at.exception, [x.message for x in at.exception]
    assert [tab.label for tab in at.tabs] == [
        "Проверить обращение",
        "Обработать CSV",
        "Качество модели",
    ]
    assert len(at.metric) == 3  # Real test report rendered, without fitting a model.
    at.text_area[0].input("My card payment appears twice in the app.").run()
    at.button[0].click().run()
    assert not at.exception, [x.message for x in at.exception]
    assert any(x.value == "transaction_charged_twice" for x in at.subheader)
    assert any(x.value == "Похожие обращения из train_fit" for x in at.subheader)
    run_id, _, report = run_paths(load_config())
    write_json(
        report / "ui_smoke.json",
        {
            "run_id": run_id,
            "mode": "smoke",
            "created_at": now(),
            "status": "passed",
            "checks": [
                "three_russian_tabs",
                "real_test_metrics",
                "interactive_predict",
                "similar_examples",
                "no_streamlit_exceptions",
            ],
            "csv_note": "CSV exercised by smoke_http.py through shared csv_io helpers",
        },
    )
    print(f"Streamlit interactive smoke passed: {run_id}")


if __name__ == "__main__":
    main()
