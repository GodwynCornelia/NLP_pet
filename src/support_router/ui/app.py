import os

import httpx
import pandas as pd
import streamlit as st

from support_router.config import load_config
from support_router.csv_io import csv_batches, export_csv, read_upload, results_frame

st.set_page_config(page_title="Support Router", page_icon="↗", layout="wide")
API = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
LIMITS = load_config()["limits"]
st.title("Support Router")
st.caption("Маршрутизация банковских обращений • BANKING77 • локальный CPU")
st.info(
    "Модель обрабатывает обращения на английском. Confidence — некалиброванная оценка модели; "
    "это не гарантия правильности. Сервис не обнаруживает неизвестные темы."
)


def call(method, path, **kwargs):
    with httpx.Client(timeout=120) as client:
        response = client.request(method, API + path, **kwargs)
        response.raise_for_status()
        return response.json()


try:
    metadata = call("GET", "/metadata")
except (httpx.HTTPError, ValueError):
    st.warning("API пока не готов. Запустите bootstrap и API; проверьте /ready и docs/progress.md.")
    st.stop()

st.caption(f"Запуск: {metadata['run_id']} · Модель: {metadata['model']}")
single, batch, quality = st.tabs(["Проверить обращение", "Обработать CSV", "Качество модели"])
with single:
    examples = [
        "My card payment appears twice in the app.",
        "I cannot withdraw cash from the ATM.",
        "I need to change the address on my account.",
    ]
    example = st.selectbox("Пример обращения", ["Свой текст"] + examples)
    text = st.text_area(
        "Обращение на английском",
        value="" if example == "Свой текст" else example,
        max_chars=LIMITS["max_text_chars"],
        height=130,
    )
    if st.button("Проверить", type="primary", disabled=not text.strip()):
        try:
            with st.spinner("Классифицируем и ищем похожие обращения…"):
                result = call("POST", "/predict", json={"text": text})
            st.subheader(result["predicted_category"])
            if result["routing_decision"] == "auto_route":
                st.success("Автоматическая маршрутизация")
            else:
                st.warning("Нужна ручная проверка")
                st.caption(
                    "Автоматическая маршрутизация выключена"
                    if result["review_reason"] == "auto_routing_disabled"
                    else "Confidence ниже выбранного порога"
                )
            st.dataframe(pd.DataFrame(result["top_predictions"]), hide_index=True)
            st.caption(
                f"Confidence: {result['confidence']:.4f} · Порог: {result['threshold']} · "
                f"Обработка на сервере: {result['latency_ms']:.1f} мс"
            )
            if result["warnings"]:
                st.warning(
                    "Текст усечён при вычислении эмбеддинга: модель учитывает только начало."
                )
            st.subheader("Похожие обращения из train_fit")
            st.caption("Контекст для оператора. Сходство не объясняет решение классификатора.")
            st.dataframe(pd.DataFrame(result["similar_examples"]), hide_index=True)
        except (httpx.HTTPError, ValueError):
            st.error("Не удалось получить ответ API. Проверьте готовность сервиса.")

with batch:
    st.write("UTF-8 CSV с колонкой text; id необязателен. До 5 MiB и 1000 строк.")
    upload = st.file_uploader("Выберите CSV", type=["csv"], max_upload_size=5)
    if upload is not None:
        try:
            frame = read_upload(upload.getvalue(), LIMITS)
            st.dataframe(frame.head(10), hide_index=True)
            st.caption(f"Строк: {len(frame)}. Ошибочные обращения останутся в результате.")
            if st.button("Обработать CSV"):
                progress = st.progress(0.0)
                results = []
                with st.spinner("Обрабатываем пакетами до 100 строк…"):
                    for items in csv_batches(frame, LIMITS["max_batch"]):
                        results.extend(
                            call(
                                "POST",
                                "/predict-batch",
                                json={"items": items, "include_similar": False},
                            )
                        )
                        progress.progress(len(results) / len(frame))
                output = results_frame(frame, results)
                st.dataframe(output, hide_index=True)
                st.download_button(
                    "Скачать результат",
                    export_csv(output),
                    "support_router_results.csv",
                    "text/csv",
                )
                st.caption(
                    "В экспортируемых текстовых ячейках перед формулоподобными значениями "
                    "добавляется апостроф. Исходный текст при классификации не меняется."
                )
        except ValueError as exc:
            st.error(str(exc))
        except httpx.HTTPError:
            st.error("Ошибка API при обработке CSV. Файл не сохраняется на диск.")

with quality:
    try:
        reports = call("GET", "/reports")
        st.subheader("Validation: сравнение до просмотра test")
        st.dataframe(pd.DataFrame(reports["validation"]), hide_index=True)
        test = reports["test"]
        st.subheader(f"Официальный test · {test['n']} обращений")
        a, b, c = st.columns(3)
        a.metric("Macro-F1", f"{test['macro_f1']:.4f}")
        b.metric("Accuracy", f"{test['accuracy']:.4f}")
        c.metric("Top-3 accuracy", f"{test['top3_accuracy']:.4f}")
        routing = test["routing"]
        st.write(
            f"Coverage: {routing['coverage']:.1%} · Ручная проверка: {routing['manual_review_rate']:.1%}"
        )
        if routing["accepted_count"]:
            lo, hi = routing["accepted_accuracy_wilson95"]
            st.write(
                f"Точность принятых: {routing['accepted_accuracy']:.1%}; "
                f"n={routing['accepted_count']}; интервал Уилсона 95%: [{lo:.1%}, {hi:.1%}]"
            )
            st.caption("Интервал относится к этой выборке и не является гарантией в эксплуатации.")
        else:
            st.write("Нет автоматически принятых обращений")
        st.subheader("Частые путаницы категорий")
        st.dataframe(pd.DataFrame(reports["confusions"]), hide_index=True)
        st.subheader("Автоматическая подборка ошибок для самостоятельного разбора")
        st.dataframe(pd.DataFrame(reports["errors"]), hide_index=True)
    except (httpx.HTTPError, ValueError):
        st.warning("Итоговый отчёт ещё не готов. Выполните evaluate для текущего запуска.")
