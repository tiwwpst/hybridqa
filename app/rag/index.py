from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import FakeEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import Settings

logger = logging.getLogger(__name__)


def _safe_index_id(table_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_\\-]+", "_", table_id).strip("_")
    if safe:
        return safe
    return hashlib.sha1(table_id.encode("utf-8")).hexdigest()[:16]


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []
    if chunk_size <= 0:
        return [text]
    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size]
        if chunk:
            chunks.append(chunk)
    return chunks


def _embedding_id(settings: Settings) -> str:
    if settings.use_fake_embeddings:
        return "fake"
    if settings.embeddings_provider == "openai":
        return settings.embeddings_model
    return settings.hf_embeddings_model


def _embedding_dim(embeddings) -> Optional[int]:
    try:
        return len(embeddings.embed_query("dimension probe"))
    except Exception:
        return None


@lru_cache(maxsize=4)
def _cached_embeddings(
    use_fake: bool,
    provider: str,
    embeddings_model: str,
    hf_model: str,
    openai_key: str,
):
    if use_fake:
        return FakeEmbeddings(size=768)
    if provider == "openai" and openai_key:
        os.environ.setdefault("OPENAI_API_KEY", openai_key)
        return OpenAIEmbeddings(model=embeddings_model)
    return HuggingFaceEmbeddings(model_name=hf_model)


def select_embeddings(settings: Settings):
    return _cached_embeddings(
        settings.use_fake_embeddings,
        settings.embeddings_provider,
        settings.embeddings_model,
        settings.hf_embeddings_model,
        settings.openai_api_key or "",
    )


def build_documents(
    table_texts: Iterable[object],
    passages: list[dict],
    settings: Settings,
    table_id: str,
) -> list[Document]:
    docs: list[Document] = []
    for item in table_texts:
        meta = {}
        text = ""
        if isinstance(item, dict):
            text = str(item.get("text", ""))
            meta = dict(item.get("metadata", {}))
        elif isinstance(item, tuple) and len(item) == 2:
            text = str(item[0])
            meta = dict(item[1] or {})
        else:
            text = str(item)
        meta.setdefault("source", "table")
        meta.setdefault("source_type", "row")
        meta.setdefault("table_id", table_id)
        docs.append(Document(page_content=text, metadata=meta))
    for passage in passages:
        content = passage.get("text") or passage.get("content") or ""
        chunks = _chunk_text(content, settings.rag_chunk_size, settings.rag_chunk_overlap)
        for chunk_id, chunk in enumerate(chunks):
            meta = {
                "source": "passage",
                "source_type": "passage_chunk",
                "chunk_id": chunk_id,
                "title": passage.get("title"),
                "url": passage.get("url"),
                "table_id": table_id,
            }
            docs.append(Document(page_content=chunk, metadata=meta))
    return docs


def build_or_load_vectorstore(
    table_id: str,
    docs: list[Document],
    index_dir: Path,
    settings: Settings,
) -> FAISS:
    embeddings = select_embeddings(settings)
    index_path = index_dir / _safe_index_id(table_id)
    meta_path = index_path / "meta.json"
    meta = {
        "table_id": table_id,
        "embedding_model_id": _embedding_id(settings),
        "embedding_dim": _embedding_dim(embeddings),
        "chunk_size": settings.rag_chunk_size,
        "chunk_overlap": settings.rag_chunk_overlap,
    }
    if index_path.exists():
        try:
            if meta_path.exists():
                existing = json.loads(meta_path.read_text(encoding="utf-8"))
                if existing == meta:
                    return FAISS.load_local(
                        str(index_path),
                        embeddings,
                        allow_dangerous_deserialization=True,
                    )
            shutil.rmtree(index_path)
        except Exception as exc:
            logger.warning("Failed to load existing index, rebuilding: %s", exc)
            shutil.rmtree(index_path, ignore_errors=True)
    vectorstore = FAISS.from_documents(docs, embeddings)
    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    meta_path.write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")
    return vectorstore


def rag_search(
    table_id: str,
    table_texts: Iterable[object],
    passages: list[dict],
    settings: Settings,
    query: str,
    k: int,
) -> list[Document]:
    docs = build_documents(table_texts, passages, settings, table_id)
    if not docs:
        return []
    vectorstore = build_or_load_vectorstore(table_id, docs, Path(settings.index_dir), settings)
    return vectorstore.similarity_search(query, k=k)
