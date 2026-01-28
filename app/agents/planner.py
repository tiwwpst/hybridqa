from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from app.agents.llm import build_llm
from app.agents.utils import extract_json
from app.config import Settings


def plan_question(question: str, table_schema: str, settings: Settings) -> dict:
    fallback = {
        "steps": [
            "Inspect the table schema and identify relevant columns",
            "Locate the target row(s) in the table",
            "Use RAG lookup for any missing context",
            "Compose final answer and JSON summary",
        ],
        "need_rag": True,
        "notes": "fallback planner",
    }

    llm = build_llm(settings, settings.openai_model_planner)
    if llm is None:
        return fallback

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are Planner-LLM. Break down the question into ReAct-style steps. "
                    "Return ONLY JSON with keys: steps (list of strings), need_rag (bool), notes (string)."
                )
            ),
            HumanMessage(
                content=(
                    f"Question: {question}\n"
                    f"Table schema: {table_schema}\n"
                    "Respond with JSON only."
                )
            ),
        ]
    )
    try:
        response = llm.invoke(prompt.format_messages())
    except Exception:
        return fallback
    try:
        payload = extract_json(response.content)
    except Exception:
        return fallback
    if not isinstance(payload, dict):
        return fallback
    return payload
