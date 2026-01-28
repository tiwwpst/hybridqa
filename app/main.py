from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder

from app.agents.graph import run_agent
from app.config import settings
from app.models import AnswerPayload, AskRequest, AskResponse, FunctionCall

app = FastAPI(title="HybridQA Multi-Agent API")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


def _validate_table_id(table_id: str) -> None:
    if ".." in table_id or "/" in table_id or "\\" in table_id:
        raise HTTPException(status_code=400, detail="invalid table_id")
    if settings.table_id_pattern and not re.match(settings.table_id_pattern, table_id):
        raise HTTPException(status_code=400, detail="table_id does not match allowed pattern")


def _validate_table_path(table_path: str) -> None:
    if not settings.allow_table_path:
        raise HTTPException(status_code=400, detail="table_path is disabled")
    root = Path(settings.table_path_root or settings.data_dir).resolve()
    resolved = Path(table_path).expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="table_path outside allowed root") from exc


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="question is required")
    if request.table_id:
        _validate_table_id(request.table_id)
    if request.table_path:
        _validate_table_path(request.table_path)
    try:
        result = run_agent(
            question=request.question,
            table_id=request.table_id,
            table_path=request.table_path,
            use_sample=request.use_sample,
            rag_k=request.rag_k,
            debug=request.debug,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    function_calls = [FunctionCall(**call) for call in result.get("function_calls", [])]
    payload = AnswerPayload(
        final_answer=result.get("final_answer", ""),
        final_answer_structured=result.get("final_answer_structured"),
        reasoning=result.get("reasoning", ""),
        function_calls=function_calls,
        sources=result.get("sources", []),
        rag_debug=result.get("rag_debug") if request.debug else None,
    )
    response = AskResponse(question=request.question, table_id=result.get("table_id"), payload=payload)
    return jsonable_encoder(response)
