from __future__ import annotations

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.llm import build_llm
from app.agents.tools import ToolCallTracker, build_table_tools
from app.agents.utils import extract_json
from app.config import Settings


def _is_substring_query(question: str) -> bool:
    lowered = question.lower()
    keywords = [
        "contains",
        "contain",
        "substring",
        "not exact match",
        "подстрока",
        "встречается",
        "содержит",
    ]
    return any(token in lowered for token in keywords)


def _tools_description(tools) -> str:
    lines = []
    for tool in tools:
        lines.append(f"- {tool.name}: {tool.description}")
    return "\n".join(lines)


def run_table_agent(
    question: str,
    table_schema: str,
    df,
    settings: Settings,
    tracker: ToolCallTracker,
    plan_steps: Optional[list[str]] = None,
) -> str:
    fallback = df.head(3).to_dict(orient="records")
    llm = build_llm(settings, settings.openai_model_table)
    if llm is None:
        return f"LLM unavailable; showing sample rows: {fallback}"

    tools = build_table_tools(df, tracker)
    tools_desc = _tools_description(tools)
    substring_query = _is_substring_query(question)
    table_required = True
    table_tool_names = {tool.name for tool in tools}
    tool_retry = False

    system = SystemMessage(
        content=(
            "You are Table-Tool Agent. Use tools to answer questions about the table. "
            "Rules: substring/contains => use filter_rows_contains (it returns row_index); "
            "exact equality => use find_rows_exact. "
            "For substring queries, follow: get_schema -> choose name column -> filter_rows_contains "
            "-> use row_index to fetch names if needed. "
            "Row indices are 0-based. For 'first row/person/item' use get_row(0) or lookup_cell(row_index=0,...). "
            "Do NOT answer without calling at least one table tool when the question is about the table. "
            "Respond ONLY in JSON. Either {\"action\": tool_name, \"args\": {...}} "
            "or {\"final\": \"answer\"}."
        )
    )
    user = HumanMessage(
        content=(
            f"Question: {question}\n"
            f"Table schema: {table_schema}\n"
            f"Plan steps: {plan_steps or []}\n"
            f"Tools:\n{tools_desc}\n"
            "Return JSON only."
        )
    )

    messages = [system, user]
    for _ in range(4):
        try:
            response = llm.invoke(messages)
        except Exception as exc:
            return f"LLM error: {exc}"
        try:
            payload = extract_json(response.content)
        except Exception:
            return f"Unable to parse tool action. Sample rows: {fallback}"
        if not isinstance(payload, dict):
            return f"Invalid tool payload. Sample rows: {fallback}"
        if "final" in payload:
            if table_required and not any(
                call.get("name") in table_tool_names for call in tracker.calls
            ):
                if not tool_retry:
                    tool_retry = True
                    messages.append(
                        HumanMessage(
                            content=(
                                "You must call at least one table tool before answering. "
                                "Return JSON action for a table tool."
                            )
                        )
                    )
                    continue
                return "insufficient_tool_evidence"
            if substring_query and not any(
                call.get("name") == "filter_rows_contains" for call in tracker.calls
            ):
                messages.append(
                    HumanMessage(
                        content=(
                            "Substring/contains query: you MUST call filter_rows_contains "
                            "before answering. Return JSON action for filter_rows_contains."
                        )
                    )
                )
                continue
            return str(payload["final"])
        action = payload.get("action")
        args = payload.get("args", {})
        if not action:
            return f"No action returned. Sample rows: {fallback}"
        if action == "find_rows":
            action = "find_rows_exact"
        if action == "find_rows_exact" and not args.get("value"):
            messages.append(
                HumanMessage(
                    content="find_rows_exact requires a non-empty value. Choose the exact value."
                )
            )
            continue
        if substring_query and action == "find_rows_exact":
            column = args.get("column")
            substring = args.get("substring") or args.get("value")
            if column and substring is not None:
                action = "filter_rows_contains"
                args = {"column": column, "substring": substring}
            else:
                messages.append(
                    HumanMessage(
                        content=(
                            "Substring query detected. Use filter_rows_contains with "
                            "column and substring."
                        )
                    )
                )
                continue
        tool = next((t for t in tools if t.name == action), None)
        if tool is None:
            return f"Unknown tool action '{action}'. Sample rows: {fallback}"
        try:
            observation = tool.invoke(args)
        except Exception as exc:
            observation = f"tool_error: {exc}"
        messages.append(
            HumanMessage(content=f"Observation from {action}: {observation}")
        )

    return f"Tool loop limit reached. Sample rows: {fallback}"
