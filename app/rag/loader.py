from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

def _candidate_table_paths(table_id: str, data_dir: Path) -> list[Path]:
    return [
        data_dir / "tables" / f"{table_id}.csv",
        data_dir / "tables" / f"{table_id}.json",
        data_dir / "tables_tok" / f"{table_id}.json",
        data_dir / "WikiTables-WithLinks" / "tables_tok" / f"{table_id}.json",
    ]


def _candidate_passage_paths(table_id: str, data_dir: Path) -> list[Path]:
    return [
        data_dir / "passages" / f"{table_id}.json",
        data_dir / "request_tok" / f"{table_id}.json",
        data_dir / "WikiTables-WithLinks" / "request_tok" / f"{table_id}.json",
    ]


def resolve_table_path(
    table_id: Optional[str],
    table_path: Optional[str],
    data_dir: Path,
    sample_dir: Path,
    use_sample: bool,
) -> Path:
    if table_path:
        return Path(table_path)
    if use_sample:
        return sample_dir / "tables" / "sample_table.csv"
    if not table_id:
        raise ValueError("table_id or table_path is required unless use_sample is true")
    for path in _candidate_table_paths(table_id, data_dir):
        if path.exists():
            return path
    raise FileNotFoundError(f"table not found for table_id={table_id}")


def _normalize_links(links: object) -> list[str]:
    if not links:
        return []
    if isinstance(links, list):
        return [str(item) for item in links if item]
    return [str(links)]


def _parse_cell(cell: object) -> tuple[object, list[str]]:
    if isinstance(cell, list):
        value = cell[0] if cell else None
        links = _normalize_links(cell[1]) if len(cell) > 1 else []
        return value, links
    return cell, []


def _parse_wikitables_json(payload: dict) -> Optional[pd.DataFrame]:
    if not isinstance(payload, dict) or "header" not in payload or "data" not in payload:
        return None
    headers = []
    header_links: dict[str, list[str]] = {}
    for header in payload.get("header", []):
        if isinstance(header, list):
            name = str(header[0]) if header else ""
            links = _normalize_links(header[1]) if len(header) > 1 else []
        else:
            name = str(header)
            links = []
        headers.append(name)
        if links:
            header_links[name] = links
    rows = []
    cell_links: dict[tuple[int, str], list[str]] = {}
    for row_index, row in enumerate(payload.get("data", [])):
        values = []
        for col_index, cell in enumerate(row):
            value, links = _parse_cell(cell)
            values.append(value)
            if links:
                column = headers[col_index] if col_index < len(headers) else str(col_index)
                cell_links[(row_index, column)] = links
        rows.append(values)
    df = pd.DataFrame(rows, columns=headers)
    df.attrs["cell_links"] = cell_links
    df.attrs["header_links"] = header_links
    return df


def load_table(
    table_id: Optional[str],
    table_path: Optional[str],
    data_dir: Path,
    sample_dir: Path,
    use_sample: bool,
) -> tuple[str, pd.DataFrame]:
    path = resolve_table_path(table_id, table_path, data_dir, sample_dir, use_sample)
    resolved_table_id = table_id or path.stem
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        df = _parse_wikitables_json(payload)
        if df is None:
            df = pd.read_json(path)
    else:
        raise ValueError("unsupported table format; use .csv or .json")
    df.attrs.setdefault("cell_links", {})
    df.attrs.setdefault("header_links", {})
    return resolved_table_id, df


def get_cell_links(df: pd.DataFrame, row_index: int, column: str) -> list[str]:
    links = df.attrs.get("cell_links", {})
    return list(links.get((row_index, column), []))


def get_row_links(df: pd.DataFrame, row_index: int) -> list[str]:
    links = df.attrs.get("cell_links", {})
    collected: list[str] = []
    seen = set()
    for (row, _column), values in links.items():
        if row != row_index:
            continue
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            collected.append(value)
    return collected


def _title_from_url(url: str) -> str:
    match = re.search(r"/wiki/(.+)", url)
    if not match:
        return url
    title = match.group(1).replace("_", " ")
    return title


def load_passages(
    table_id: str,
    data_dir: Path,
    sample_dir: Path,
    use_sample: bool,
) -> list[dict]:
    if use_sample:
        path = sample_dir / "passages" / "sample_passages.json"
    else:
        path = None
        for candidate in _candidate_passage_paths(table_id, data_dir):
            if candidate.exists():
                path = candidate
                break
        if path is None:
            logger.warning("No passages found for table_id=%s under %s", table_id, data_dir)
            return []
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        passages = []
        for url, text in payload.items():
            passages.append(
                {
                    "url": url,
                    "title": _title_from_url(url),
                    "text": text,
                }
            )
        return passages
    return []


def iter_table_rows_as_text(df: pd.DataFrame) -> Iterable[str]:
    for idx, row in df.iterrows():
        parts = [f"{col}={row[col]}" for col in df.columns]
        yield f"row_{idx}: " + ", ".join(parts)


def table_schema_text(df: pd.DataFrame) -> str:
    columns = ", ".join(str(col) for col in df.columns)
    return f"columns: {columns}; rows: {len(df)}"
