from __future__ import annotations

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from app.agents.llm import build_llm
from app.agents.utils import extract_json
from app.config import Settings


def run_analysis_agent(
    question: str,
    table_answer: str,
    rag_context: list[str],
    settings: Settings,
    plan_steps: Optional[list[str]] = None,
) -> dict:
    fallback = {
        "final_answer": table_answer or "No answer produced.",
        "reasoning": "Fallback reasoning used.",
        "error": "analysis_fallback",
    }
    llm = build_llm(settings, settings.openai_model_analysis)
    if llm is None:
        return fallback

    context_text = "\n".join(rag_context[: settings.max_rag_results])

    def _attempt(strict: bool) -> Optional[dict]:
        system_rules = (
            "You are Analysis-Agent. Combine table evidence and RAG context. "
            "Return ONLY JSON with keys: final_answer, reasoning. "
            "final_answer MUST be non-empty. "
            "Do NOT claim the table contains facts that are only in RAG. "
            "You MUST answer both parts of the question."
        )
        if strict:
            system_rules += " Do not add any extra keys or commentary."
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_rules),
                HumanMessage(
                    content=(
                        f"Question: {question}\n"
                        f"Plan steps: {plan_steps or []}\n"
                        f"Table answer: {table_answer}\n"
                        f"RAG context: {context_text}\n"
                        "Return JSON only."
                    )
                ),
            ]
        )
        try:
            response = llm.invoke(prompt.format_messages())
        except Exception as exc:
            fallback["error"] = f"analysis_llm_error: {exc}"
            return None
        try:
            payload = extract_json(response.content)
        except Exception:
            fallback["error"] = "analysis_invalid_json"
            return None
        if not isinstance(payload, dict):
            fallback["error"] = "analysis_non_dict_json"
            return None
        if not payload.get("final_answer") or not payload.get("reasoning"):
            fallback["error"] = "analysis_missing_fields"
            return None
        return payload

    payload = _attempt(strict=False)
    if payload is None:
        payload = _attempt(strict=True)
    if payload is None:
        return fallback
    return payload
