import io
import json

import pandas as pd


def read_upload(content, limits):
    if len(content) > limits["csv_bytes"]:
        raise ValueError("CSV превышает 5 MiB")
    try:
        text = content.decode("utf-8-sig", errors="strict")
        frame = pd.read_csv(
            io.StringIO(text),
            dtype=str,
            keep_default_na=False,
            skip_blank_lines=False,
            nrows=limits["csv_rows"] + 1,
        )
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("Нужен корректный CSV в UTF-8") from exc
    if "text" not in frame.columns:
        raise ValueError("В CSV обязательна колонка text")
    if not 1 <= len(frame) <= limits["csv_rows"]:
        raise ValueError("CSV должен содержать 1..1000 строк")
    return frame


def csv_batches(frame, max_batch=100):
    for start in range(0, len(frame), max_batch):
        yield [
            {"id": f"row-{i + 1:06d}", "text": frame.iloc[i]["text"]}
            for i in range(start, min(start + max_batch, len(frame)))
        ]


def results_frame(frame, results):
    expected = [f"row-{i + 1:06d}" for i in range(len(frame))]
    if [row["id"] for row in results] != expected:
        raise ValueError("API changed CSV row count or order")
    output = frame.copy()
    # Reserved output names replace same-named uploaded columns; input id/text are retained.
    output["row_id"] = expected
    for key in ["predicted_category", "confidence", "routing_decision"]:
        output[key] = [(row["result"] or {}).get(key, "") for row in results]
    output["top3"] = [
        json.dumps((row["result"] or {}).get("top_predictions", []), ensure_ascii=False)
        for row in results
    ]
    output["error"] = [
        json.dumps(row["error"], ensure_ascii=False) if row["error"] else "" for row in results
    ]
    return output


def safe_cell(value):
    if isinstance(value, str) and value.lstrip(" \t\r\n").startswith(("=", "+", "-", "@")):
        return "'" + value
    if isinstance(value, str) and value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value


def export_csv(frame):
    return frame.map(safe_cell).to_csv(index=False).encode("utf-8-sig")
