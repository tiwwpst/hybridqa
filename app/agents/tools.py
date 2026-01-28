from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from app.config import Settings
from app.rag.index import rag_search
from app.rag.loader import get_cell_links


@dataclass
class ToolCallTracker:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def record(self, name: str, arguments: dict[str, Any]) -> None:
        self.calls.append({"name": name, "arguments": arguments})


def build_table_tools(df: pd.DataFrame, tracker: ToolCallTracker):
    @tool(description="Return table column names and row count.")
    def get_schema() -> str:
        tracker.record("get_schema", {})
        return f"columns={list(df.columns)} rows={len(df)}"

    @tool(description="Return column names.")
    def get_columns() -> list[str]:
        tracker.record("get_columns", {})
        return [str(col) for col in df.columns]

    @tool(description="Return number of rows.")
    def row_count() -> int:
        tracker.record("row_count", {})
        return int(len(df))

    @tool(description="Return a row as a dict by index.")
    def get_row(index: int) -> dict:
        tracker.record("get_row", {"index": index})
        if index < 0 or index >= len(df):
            return {}
        row = df.iloc[int(index)]
        return {col: row[col] for col in df.columns}

    @tool(description="Return a column values list.")
    def get_column(name: str) -> list:
        tracker.record("get_column", {"name": name})
        if name not in df.columns:
            return []
        return df[name].tolist()

    @tool(description="Find row indices where column exactly matches value.")
    def find_rows_exact(column: str, value: str) -> list[int]:
        tracker.record("find_rows_exact", {"column": column, "value": value})
        if column not in df.columns:
            return []
        matches = df[df[column].astype(str) == str(value)]
        return matches.index.tolist()

    @tool(description="Return rows (with row_index) where column contains substring (case-insensitive).")
    def filter_rows_contains(column: str, substring: str) -> list[dict]:
        tracker.record("filter_rows_contains", {"column": column, "substring": substring})
        if column not in df.columns:
            return []
        mask = df[column].astype(str).str.contains(str(substring), case=False, na=False)
        rows = df[mask]
        rows = rows.reset_index().rename(columns={"index": "row_index"})
        return rows.to_dict(orient="records")

    @tool(description="Return a cell by row index and column name.")
    def lookup_cell(row_index: int, column: str) -> str:
        tracker.record("lookup_cell", {"row_index": row_index, "column": column})
        if column not in df.columns:
            return ""
        if row_index < 0 or row_index >= len(df):
            return ""
        return str(df.iloc[int(row_index)][column])

    @tool(description="Return links for a table cell (if available).")
    def lookup_links(row_index: int, column: str) -> list[str]:
        tracker.record("lookup_links", {"row_index": row_index, "column": column})
        if row_index < 0 or row_index >= len(df):
            return []
        if column not in df.columns:
            return []
        return get_cell_links(df, int(row_index), column)

    return [
        get_schema,
        get_columns,
        row_count,
        get_row,
        get_column,
        find_rows_exact,
        filter_rows_contains,
        lookup_cell,
        lookup_links,
    ]


def build_rag_tool(
    table_id: str,
    df: pd.DataFrame,
    passages: list[dict],
    settings: Settings,
    tracker: ToolCallTracker,
):
    @tool(description="Retrieve relevant table rows and passages via vector search.")
    def rag_lookup(query: str, k: int = 5) -> list[str]:
        tracker.record("rag_lookup", {"query": query, "k": k})
        table_texts = []
        for idx, row in df.iterrows():
            parts = [f"{col}={row[col]}" for col in df.columns]
            table_texts.append(
                {
                    "text": f"row_{idx}: " + ", ".join(parts),
                    "metadata": {"source_type": "row", "row_index": int(idx)},
                }
            )
        for col in df.columns:
            table_texts.append(
                {
                    "text": f"column:{col}",
                    "metadata": {"source_type": "column", "column_name": str(col)},
                }
            )
        docs = rag_search(
            table_id=table_id,
            table_texts=table_texts,
            passages=passages,
            settings=settings,
            query=query,
            k=k,
        )
        return [doc.page_content for doc in docs]

    return rag_lookup
