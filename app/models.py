from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    table_id: Optional[str] = None
    table_path: Optional[str] = None
    use_sample: bool = False
    rag_k: Optional[int] = Field(default=None, ge=0)
    debug: bool = False


class FunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any]


class AnswerPayload(BaseModel):
    final_answer: str
    final_answer_structured: Optional[Any] = None
    reasoning: str
    function_calls: list[FunctionCall]
    sources: list[str] = Field(default_factory=list)
    rag_debug: Optional[list[dict[str, Any]]] = None


class AskResponse(BaseModel):
    question: str
    table_id: Optional[str]
    payload: AnswerPayload
