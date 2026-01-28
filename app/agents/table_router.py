from __future__ import annotations

import json
import re
from typing import Any, Optional

from app.agents.tools import build_table_tools


_TABLE_ONLY_TOKENS = [
    "table only",
    "use only the table",
    "using only the table",
    "only the table",
    "только таблиц",
    "используй только таблицу",
    "используйте только таблицу",
    "только таблицу",
    "только по таблице",
]


def detect_table_only(question: str) -> bool:
    lowered = question.lower()
    return any(token in lowered for token in _TABLE_ONLY_TOKENS)


def _extract_quoted(text: str) -> Optional[str]:
    matches = re.findall(r"\"([^\"]+)\"|'([^']+)'", text)
    for first, second in matches:
        value = first or second
        if value:
            return value
    return None


def _match_column_in_question(question: str, columns: list[str]) -> Optional[str]:
    lowered = question.lower()
    quoted = _extract_quoted(question)
    if quoted:
        for col in columns:
            if col.lower() == quoted.lower():
                return col
    for col in sorted(columns, key=len, reverse=True):
        if col and col.lower() in lowered:
            return col
    return None


def _select_name_column(question: str, columns: list[str]) -> Optional[str]:
    lowered = question.lower()
    keyword_map = [
        ("name", "name"),
        ("player", "player"),
        ("athlete", "athlete"),
        ("runner", "runner"),
        ("driver", "driver"),
        ("person", "person"),
        ("museum", "museum"),
        ("title", "title"),
    ]
    for keyword, token in keyword_map:
        if keyword in lowered:
            for col in columns:
                if token in col.lower():
                    return col
    return None


def _select_location_column(question: str, columns: list[str]) -> Optional[str]:
    lowered = question.lower()
    tokens = ("city", "location", "place", "town", "region", "state", "area", "место", "город")
    if any(token in lowered for token in tokens):
        for col in columns:
            if any(token in col.lower() for token in tokens):
                return col
    return None


def _select_time_column(columns: list[str]) -> Optional[str]:
    for col in columns:
        if col and "time" in col.lower():
            return col
    return None


def _parse_time_value(value: object) -> Optional[float]:
    text = str(value).strip()
    if not text:
        return None
    if ":" in text:
        parts = text.split(":")
        try:
            if len(parts) == 2:
                hours = float(parts[0])
                min_sec = parts[1]
                if "." in min_sec:
                    minutes, seconds = min_sec.split(".", 1)
                else:
                    minutes, seconds = min_sec, "0"
                return hours * 3600 + float(minutes) * 60 + float(seconds)
            if len(parts) == 3:
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
        except ValueError:
            return None
    numeric = re.sub(r"[^\d.\-]", "", text)
    if not numeric:
        return None
    try:
        return float(numeric)
    except ValueError:
        return None


def _sorted_indices_by_time(df, time_col: str, ascending: bool = True) -> list[int]:
    values: list[tuple[int, float]] = []
    for idx, row in df.iterrows():
        parsed = _parse_time_value(row[time_col])
        if parsed is None:
            parsed = float("inf") if ascending else float("-inf")
        values.append((int(idx), parsed))
    if not values:
        return []
    return [idx for idx, _ in sorted(values, key=lambda item: item[1], reverse=not ascending)]


def _extract_top_n(question: str) -> int:
    lowered = question.lower()
    match = re.search(r"top\s+(\d+)", lowered)
    if match:
        return int(match.group(1))
    match = re.search(r"топ\s+(\d+)", lowered)
    if match:
        return int(match.group(1))
    return _extract_first_n(question)


def _select_metric_column(question: str, columns: list[str]) -> Optional[str]:
    lowered = question.lower()
    if "yard" in lowered:
        tokens = ("yard", "yards", "yd", "yds", "rushing")
    elif "time" in lowered:
        tokens = ("time",)
    elif "score" in lowered or "points" in lowered:
        tokens = ("score", "points", "pts")
    else:
        tokens = ("count", "total", "number")
    for col in columns:
        if any(token in col.lower() for token in tokens):
            return col
    return None


def _parse_number(value: object) -> Optional[float]:
    text = str(value).strip()
    if not text:
        return None
    cleaned = re.sub(r"[^\d.\-]", "", text)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _select_first_text_column(df, columns: list[str]) -> Optional[str]:
    for col in columns:
        sample = df[col].head(3).tolist() if col in df.columns else []
        if any(_parse_number(value) is None and str(value).strip() for value in sample):
            return col
    return columns[0] if columns else None


def _label_for_column(question: str, column: str) -> str:
    lowered = question.lower()
    if "museum" in lowered:
        return "museum_name"
    if "player" in lowered:
        return "player"
    if "person" in lowered:
        return "person"
    if "name" in lowered:
        return "name"
    return column or "column_value"


def _stringify_row(row: dict) -> dict:
    cleaned = {}
    for key, value in row.items():
        cleaned[key] = "" if value is None else str(value)
    return cleaned


def _extract_row_index(question: str) -> Optional[int]:
    lowered = question.lower()
    if any(
        token in lowered
        for token in (
            "first row",
            "first person",
            "first player",
            "first runner",
            "first driver",
            "first athlete",
            "first item",
            "first entry",
            "first record",
        )
    ):
        return 0
    if "первая строка" in lowered or "первый" in lowered:
        return 0
    ordinal_map = {
        "second": 1,
        "third": 2,
        "fourth": 3,
        "fifth": 4,
        "sixth": 5,
        "seventh": 6,
        "eighth": 7,
        "ninth": 8,
        "tenth": 9,
        "вторая": 1,
        "третья": 2,
        "четвертая": 3,
        "пятая": 4,
    }
    for word, index in ordinal_map.items():
        if any(
            token in lowered
            for token in (
                f"{word} row",
                f"{word} строк",
                f"{word} person",
                f"{word} player",
                f"{word} item",
            )
        ):
            return index
    match = re.search(r"(row|index)\s*(\d+)", lowered)
    if match:
        return int(match.group(2))
    match = re.search(r"(\d+)(st|nd|rd|th)\s+row", lowered)
    if match:
        return max(int(match.group(1)) - 1, 0)
    return None


def _extract_first_n(question: str) -> int:
    lowered = question.lower()
    match = re.search(r"first\s+(\d+)", lowered)
    if match:
        return int(match.group(1))
    match = re.search(r"первые\s+(\d+)", lowered)
    if match:
        return int(match.group(1))
    word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    for word, value in word_map.items():
        if f"first {word}" in lowered:
            return value
    return 3


def _json_intent(question: str) -> bool:
    lowered = question.lower()
    return "json" in lowered or "в json" in lowered


def _extract_json_keys(question: str) -> list[str]:
    match = re.search(r"\{([^{}]+)\}", question)
    if not match:
        return []
    body = match.group(1)
    return re.findall(r'"([^"]+)"\s*:', body)


def _build_json_value(keys: list[str], row_index: int, column: str, value: object) -> dict:
    if not keys:
        return {"row_index": row_index, "value": value}
    payload: dict[str, object] = {}
    for key in keys:
        lowered = key.lower()
        if "row" in lowered or "index" in lowered:
            payload[key] = row_index
        elif "column" in lowered:
            payload[key] = column
        elif any(token in lowered for token in ("value", "answer", "result", "name")):
            payload[key] = value
        else:
            payload[key] = value
    return payload


def route_table_question(question: str, df, tracker) -> dict:
    tools = {tool.name: tool for tool in build_table_tools(df, tracker)}
    lowered = question.lower()
    table_only = detect_table_only(question)
    columns = [str(col) for col in df.columns]
    json_intent = _json_intent(question)

    if re.search(r"how many rows|number of rows|row count|сколько строк", lowered):
        count = tools["row_count"].invoke({})
        return {
            "handled": True,
            "answer_text": str(count),
            "answer_structured": count,
            "reasoning": "Counted rows using row_count tool.",
            "disable_rag": True,
            "used_links": [],
            "router_complete": True,
        }

    columns_intent = False
    if "column names" in lowered or "названия колонок" in lowered:
        columns_intent = True
    elif "columns" in lowered or "колонки" in lowered:
        if "row indices" not in lowered and "first column" not in lowered and "row" not in lowered:
            columns_intent = True
    if columns_intent:
        cols = tools["get_columns"].invoke({})
        structured = {"columns": cols}
        return {
            "handled": True,
            "answer_text": json.dumps(structured, ensure_ascii=True),
            "answer_structured": structured,
            "reasoning": "Returned column names using get_columns tool.",
            "disable_rag": True,
            "used_links": [],
            "router_complete": True,
        }

    if "row" in lowered and json_intent and ("object" in lowered or "all columns" in lowered):
        row_index = _extract_row_index(question)
        if row_index is not None:
            row = tools["get_row"].invoke({"index": row_index})
            structured = _stringify_row(row) if row else {}
            return {
                "handled": True,
                "answer_text": json.dumps(structured, ensure_ascii=True),
                "answer_structured": structured,
                "reasoning": f"Returned row {row_index} as a JSON object.",
                "disable_rag": True,
                "used_links": [],
                "router_complete": True,
            }

    if "sort" in lowered and ("time" in lowered or "times" in lowered):
        time_col = _select_time_column(columns)
        if time_col:
            ascending = "desc" not in lowered and "descending" not in lowered
            indices = _sorted_indices_by_time(df, time_col, ascending=ascending)
            n = _extract_first_n(question)
            indices = indices[:n]
            rows = []
            for idx in indices:
                row = tools["get_row"].invoke({"index": idx})
                if row:
                    rows.append(_stringify_row(row))
            structured = rows
            answer_text = json.dumps(structured, ensure_ascii=True)
            return {
                "handled": True,
                "answer_text": answer_text,
                "answer_structured": structured,
                "reasoning": f"Sorted by {time_col} ({'ascending' if ascending else 'descending'}) and returned first {len(rows)} rows.",
                "disable_rag": True,
                "used_links": [],
                "router_complete": True,
            }

    if "top" in lowered or "top " in lowered:
        n = _extract_top_n(question)
        time_col = _select_time_column(columns)
        name_col = _select_name_column(question, columns) or _select_first_text_column(df, columns)
        indices = list(df.index)
        if time_col:
            indices = _sorted_indices_by_time(df, time_col, ascending=True)
        indices = indices[:n]
        structured = []
        for idx in indices:
            row = tools["get_row"].invoke({"index": idx})
            if not row:
                continue
            row = _stringify_row(row)
            item = {"row_index": idx}
            if name_col:
                item[name_col] = row.get(name_col)
            if time_col:
                item[time_col] = row.get(time_col)
            structured.append(item)
        answer_text = json.dumps(structured, ensure_ascii=True) if json_intent else str(structured)
        return {
            "handled": True,
            "answer_text": answer_text,
            "answer_structured": structured,
            "reasoning": f"Selected top {len(structured)} rows using {time_col or 'table order'}.",
            "disable_rag": True,
            "used_links": [],
            "router_complete": True,
        }

    if any(token in lowered for token in ("most", "highest", "maximum", "max", "least", "lowest", "minimum", "min")):
        metric_col = _select_metric_column(question, columns)
        if metric_col:
            values: list[tuple[int, float]] = []
            for idx, row in df.iterrows():
                parsed = _parse_number(row[metric_col])
                if parsed is not None:
                    values.append((int(idx), parsed))
            if values:
                wants_min = any(token in lowered for token in ("least", "lowest", "minimum", "min"))
                best_idx, _ = min(values, key=lambda item: item[1]) if wants_min else max(values, key=lambda item: item[1])
                row = tools["get_row"].invoke({"index": best_idx})
                row = _stringify_row(row) if row else {}
                name_col = _select_name_column(question, columns) or _select_first_text_column(df, columns)
                name_value = row.get(name_col) if name_col and row else ""
                metric_value = row.get(metric_col) if row else ""
                structured = {"name": name_value, "value": metric_value, "row_index": best_idx}
                answer_text = json.dumps(structured, ensure_ascii=True) if json_intent else (
                    f"{name_value} had the {'least' if wants_min else 'most'} {metric_col} with {metric_value}."
                )
                return {
                    "handled": True,
                    "answer_text": answer_text,
                    "answer_structured": structured if json_intent else None,
                    "reasoning": f"Selected row {best_idx} with {'minimum' if wants_min else 'maximum'} {metric_col}.",
                    "disable_rag": True,
                    "used_links": [],
                    "router_complete": True,
                }

    city_link_intent = False
    if "linked in the first row" in lowered or "linked in first row" in lowered:
        city_link_intent = True
    elif "linked" in lowered and any(
        token in lowered
        for token in ("city", "location", "place", "where", "located", "город", "ссылка")
    ):
        city_link_intent = True
    if city_link_intent:
        row_index = _extract_row_index(question)
        if row_index is not None:
            column = _match_column_in_question(question, columns)
            if column is None:
                column = _select_location_column(question, columns)
            links: list[str] = []
            if column is None:
                for candidate in columns:
                    candidate_links = tools["lookup_links"].invoke(
                        {"row_index": row_index, "column": candidate}
                    )
                    if candidate_links:
                        column = candidate
                        links = candidate_links
                        break
            if column is None and columns:
                column = columns[0]
            if column and not links:
                links = tools["lookup_links"].invoke({"row_index": row_index, "column": column})
            value = ""
            if column:
                value = tools["lookup_cell"].invoke({"row_index": row_index, "column": column})
            used_links = [link for link in links if link]
            return {
                "handled": True,
                "answer_text": str(value),
                "answer_structured": {"value": value, "links": used_links},
                "reasoning": "Selected a linked cell from the row for follow-up RAG.",
                "disable_rag": table_only,
                "used_links": used_links,
                "router_complete": False,
            }

    if re.search(r"\blink", lowered) or any(token in lowered for token in ("wiki links", "links", "url", "ссылк")):
        row_index = _extract_row_index(question)
        column = _match_column_in_question(question, columns)
        if column is None:
            column = _select_name_column(question, columns)
        if column is None and columns:
            column = columns[0]
        if row_index is not None and column:
            links = tools["lookup_links"].invoke({"row_index": row_index, "column": column})
            used_links = [link for link in links if link]
            structured = {"links": used_links}
            return {
                "handled": True,
                "answer_text": json.dumps(structured, ensure_ascii=True),
                "answer_structured": structured,
                "reasoning": f"Looked up links for row {row_index} column '{column}'.",
                "disable_rag": True,
                "used_links": used_links,
                "router_complete": True,
            }

    if "contains" in lowered or "подстрок" in lowered or "содерж" in lowered:
        substring = _extract_quoted(question)
        if not substring:
            match = re.search(r"contains\s+([^\s.,]+)", lowered)
            substring = match.group(1) if match else None
        if substring:
            column = _match_column_in_question(question, columns)
            if column is None:
                column = _select_name_column(question, columns)
            if column is None and columns:
                column = columns[0]
            rows = tools["filter_rows_contains"].invoke(
                {"column": column, "substring": substring}
            )
            label = _label_for_column(question, column or "column_value")
            structured = []
            for row in rows:
                structured.append({"row_index": row.get("row_index"), label: row.get(column)})
            return {
                "handled": True,
                "answer_text": json.dumps(structured, ensure_ascii=True),
                "answer_structured": structured,
                "reasoning": f"Filtered rows where {column} contains '{substring}'.",
                "disable_rag": True,
                "used_links": [],
                "router_complete": True,
            }

    if "row indices" in lowered and "first column" in lowered:
        n = _extract_first_n(question)
        if columns:
            first_col = columns[0]
            tools["get_columns"].invoke({})
            values = tools["get_column"].invoke({"name": first_col})
            structured = []
            for idx, value in enumerate(values[:n]):
                structured.append({"row_index": idx, "column_value": value})
            return {
                "handled": True,
                "answer_text": json.dumps(structured, ensure_ascii=True),
                "answer_structured": structured,
                "reasoning": "Listed first rows from the first column.",
                "disable_rag": True,
                "used_links": [],
                "router_complete": True,
            }

    row_index = _extract_row_index(question)
    if row_index is not None:
        column = _match_column_in_question(question, columns)
        if column is None:
            column = _select_name_column(question, columns)
        if column is None and columns:
            column = columns[0]
        if column:
            value = tools["lookup_cell"].invoke({"row_index": row_index, "column": column})
            links = tools["lookup_links"].invoke({"row_index": row_index, "column": column})
            used_links = [link for link in links if link]
            if json_intent:
                keys = _extract_json_keys(question)
                structured = _build_json_value(keys, row_index, column, value)
                return {
                    "handled": True,
                    "answer_text": json.dumps(structured, ensure_ascii=True),
                    "answer_structured": structured,
                    "reasoning": f"Looked up row {row_index} column '{column}'.",
                    "disable_rag": True,
                    "used_links": used_links,
                    "router_complete": True,
                }
            return {
                "handled": True,
                "answer_text": str(value),
                "answer_structured": None,
                "reasoning": f"Looked up row {row_index} column '{column}'.",
                "disable_rag": True,
                "used_links": used_links,
                "router_complete": True,
            }

    return {
        "handled": False,
        "answer_text": "",
        "answer_structured": None,
        "reasoning": "",
        "disable_rag": table_only,
        "used_links": [],
        "router_complete": False,
    }
