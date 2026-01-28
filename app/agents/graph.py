from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional, TypedDict

import pandas as pd
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from app.agents.analysis_agent import run_analysis_agent
from app.agents.planner import plan_question
from app.agents.table_agent import run_table_agent
from app.agents.table_router import detect_table_only, route_table_question
from app.agents.tools import ToolCallTracker
from app.agents.utils import extract_json
from app.config import settings
from app.rag.index import rag_search
from app.rag.loader import get_cell_links, load_passages, load_table, table_schema_text


class AgentState(TypedDict, total=False):
    question: str
    table_id: Optional[str]
    table_path: Optional[str]
    use_sample: bool
    rag_k: Optional[int]
    debug: bool
    table_only: bool
    df: pd.DataFrame
    passages: list[dict]
    passages_loaded: bool
    table_schema: str
    plan_steps: list[str]
    need_rag: bool
    table_answer: str
    table_answer_structured: Optional[object]
    table_reasoning: str
    table_agent_mode: str
    router_complete: bool
    table_used_links: list[str]
    rag_debug: list[dict[str, object]]
    table_links: list[str]
    rag_docs: list[Document]
    rag_docs_by_hop: dict[int, list[Document]]
    rag_context: list[str]
    sources: list[str]
    rag_hops: list[dict]
    tracker: ToolCallTracker
    function_calls: list[dict]
    final_answer: str
    final_answer_structured: Optional[object]
    reasoning: str


def _infer_row_index(question: str) -> Optional[int]:
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
            "row 0",
            "row zero",
        )
    ):
        return 0
    if "first" in lowered and "table" in lowered:
        return 0
    if "перв" in lowered and ("строк" in lowered or "таблиц" in lowered):
        return 0
    match = re.search(r"(row|index)\s*(\d+)", lowered)
    if match:
        return int(match.group(2))
    return None


def _is_numeric_string(value: str) -> bool:
    return bool(re.match(r"^\d+(\.\d+)?$", value.strip()))


def _select_entity_from_row(
    df: pd.DataFrame,
    row_index: int,
    question: str,
) -> tuple[Optional[str], list[str], Optional[str]]:
    if df.empty or row_index < 0 or row_index >= len(df):
        return None, [], None
    lowered = question.lower()
    location_tokens = (
        "city",
        "location",
        "place",
        "town",
        "region",
        "state",
        "area",
        "district",
        "where",
        "located",
        "стране",
        "город",
    )
    name_tokens = ("player", "name", "person", "athlete", "actor", "artist", "scientist")
    service_tokens = ("rank", "no", "№", "number", "year", "date", "id", "position")
    if any(token in lowered for token in ("birthplace", "birth place", "place of birth", "born", "место рождения", "родился")):
        preferred_tokens = name_tokens
    else:
        preferred_tokens = location_tokens if any(token in lowered for token in location_tokens) else name_tokens
    columns = [col for col in df.columns if col is not None]
    linked_candidates: list[tuple[str, str, list[str]]] = []
    for col in columns:
        value = df.iloc[row_index][col]
        if pd.isna(value):
            continue
        value_str = str(value).strip()
        if not value_str:
            continue
        links = get_cell_links(df, row_index, str(col))
        if links:
            linked_candidates.append((str(col), value_str, links))
    if linked_candidates:
        preferred_linked = [
            item
            for item in linked_candidates
            if any(token in item[0].lower() for token in preferred_tokens)
        ]
        ordered = preferred_linked or linked_candidates
        for col, value_str, links in ordered:
            if _is_numeric_string(value_str):
                continue
            return value_str, links, col
        col, value_str, links = ordered[0]
        return value_str, links, col
    candidate_cols = [
        col
        for col in columns
        if not any(token in str(col).lower() for token in service_tokens)
    ]
    if not candidate_cols:
        candidate_cols = columns
    prioritized_cols = [
        col
        for col in candidate_cols
        if any(token in str(col).lower() for token in preferred_tokens)
    ]
    if not prioritized_cols:
        prioritized_cols = candidate_cols
    numeric_fallback: Optional[tuple[str, list[str], str]] = None
    for col in prioritized_cols:
        value = df.iloc[row_index][col]
        if pd.isna(value):
            continue
        value_str = str(value).strip()
        links = get_cell_links(df, row_index, str(col))
        if _is_numeric_string(value_str):
            if numeric_fallback is None:
                numeric_fallback = (value_str, links, str(col))
            continue
        return value_str, links, str(col)
    if numeric_fallback is not None:
        return numeric_fallback
    return None, [], None


def _build_entity_query(question: str, entity: str) -> str:
    lowered = question.lower()
    if any(token in lowered for token in ("birth", "born", "birthplace")):
        suffix = "place of birth born in"
    elif "middle name" in lowered:
        suffix = "middle name"
    elif any(token in lowered for token in ("nationality", "country")):
        suffix = "nationality country"
    else:
        suffix = "biography"
    return f"{entity} {suffix}"


def _extract_location_from_question(question: str) -> Optional[str]:
    match = re.search(r"\bcity\s+([^?.,]+)", question, re.IGNORECASE)
    if match:
        candidate = match.group(1)
        candidate = re.split(r"\s+in\b", candidate, maxsplit=1, flags=re.IGNORECASE)[0]
        candidate = candidate.strip()
        return candidate or None
    match = re.search(r"\bгород\s+([^?.,]+)", question, re.IGNORECASE)
    if match:
        candidate = match.group(1)
        candidate = re.split(r"\s+в\b", candidate, maxsplit=1, flags=re.IGNORECASE)[0]
        candidate = candidate.strip()
        return candidate or None
    candidates = re.findall(r"\b[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)*", question)
    stop = {"What", "Which", "Who", "How", "List", "Top", "Sort", "Return"}
    candidates = [c for c in candidates if c not in stop]
    return candidates[-1] if candidates else None


def _extract_named_entity_from_question(question: str) -> Optional[str]:
    match = re.search(r"\bwho\s+(?:is|was)\s+([^?]+)", question, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        return candidate or None
    match = re.search(r"\bкто\s+такой\s+([^?]+)", question, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        return candidate or None
    return None


def _entity_mentioned_in_docs(entity: str, docs: list[Document]) -> bool:
    if not entity:
        return False
    lowered = entity.lower()
    for doc in docs:
        meta = doc.metadata or {}
        title = str(meta.get("title") or "").lower()
        url = str(meta.get("url") or "").lower()
        content = (doc.page_content or "").lower()
        if lowered in title or lowered in url or lowered in content:
            return True
    return False


def _needs_birthplace_country(question: str) -> bool:
    lowered = question.lower()
    birthplace_tokens = (
        "birthplace",
        "birth place",
        "place of birth",
        "born",
        "место рождения",
        "родился",
    )
    country_tokens = ("which country", "country is", "в какой стране", "стране")
    return any(token in lowered for token in birthplace_tokens) and any(
        token in lowered for token in country_tokens
    )


def _needs_birthplace_city_country(question: str) -> bool:
    lowered = question.lower()
    birthplace_tokens = ("birthplace", "birth place", "place of birth", "born", "место рождения", "родился")
    country_tokens = ("which country", "country is", "what country", "в какой стране", "стране")
    city_tokens = ("city", "город")
    return (
        any(token in lowered for token in birthplace_tokens)
        and any(token in lowered for token in country_tokens)
        and any(token in lowered for token in city_tokens)
    )


def _question_requires_rag(question: str) -> bool:
    lowered = question.lower()
    tokens = [
        "birthplace",
        "birth place",
        "place of birth",
        "born",
        "who is",
        "who was",
        "which country",
        "what country",
        "country is",
        "located in",
        "where is",
        "capital",
        "место рождения",
        "родился",
        "в какой стране",
        "в какой стране находится",
        "где находится",
        "столица",
    ]
    return any(token in lowered for token in tokens)


def _needs_country_only(question: str) -> bool:
    lowered = question.lower()
    if _needs_birthplace_country(question):
        return False
    country_tokens = (
        "country",
        "which country",
        "what country",
        "located in",
        "where is",
        "страна",
        "в какой стране",
        "стране",
        "где находится",
    )
    return any(token in lowered for token in country_tokens)


def _is_valid_city(candidate: str) -> bool:
    cleaned = candidate.strip()
    if len(cleaned) < 3:
        return False
    if _contains_alias_marker(cleaned):
        return False
    lowered = cleaned.lower()
    if lowered.startswith("the "):
        return False
    if cleaned.isupper():
        return False
    if any(char.isdigit() for char in cleaned):
        return False
    stop_words = {
        "nfl",
        "nba",
        "mlb",
        "nhl",
        "season",
        "team",
        "league",
        "university",
        "college",
        "present",
    }
    for word in stop_words:
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return False
    if not re.search(r"[A-Za-z]", cleaned):
        return False
    return True


def _is_valid_birthplace_string(birthplace: str) -> bool:
    if _contains_alias_marker(birthplace):
        return False
    cleaned = birthplace.strip()
    if len(cleaned) < 3:
        return False
    lowered = cleaned.lower()
    if cleaned.isupper():
        return False
    if any(char.isdigit() for char in cleaned):
        return False
    stop_words = {
        "nfl",
        "nba",
        "mlb",
        "nhl",
        "season",
        "team",
        "league",
        "university",
        "college",
        "present",
    }
    for word in stop_words:
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return False
    if not re.search(r"[A-Za-z]", cleaned):
        return False
    return True


def _contains_alias_marker(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "also known as",
        "known as",
        "nicknamed",
        "aka",
        "a.k.a",
        "alias",
        "formerly known as",
        "also called",
    )
    return any(marker in lowered for marker in markers)


def _strip_leading_in(text: str) -> str:
    return re.sub(r"^\s*in\s+", "", text, flags=re.IGNORECASE).strip()


def _strip_parentheticals(text: str) -> str:
    def replace(match: re.Match) -> str:
        content = match.group(1).strip()
        if not content:
            return ""
        if _contains_alias_marker(content):
            return ""
        if re.search(r"\d{4}", content) and not re.search(r"[A-Za-z]", content):
            return ""
        return f" {content} "

    return re.sub(r"\(([^)]+)\)", replace, text)


def _extract_birthplace_from_born_in(docs: list[Document]) -> Optional[str]:
    for doc in docs:
        text = doc.page_content or ""
        cleaned = re.sub(
            r"\([^)]*(also known as|aka|nicknamed)[^)]*\)",
            "",
            text,
            flags=re.IGNORECASE,
        )
        match = re.search(r"\bborn\s+in\s+([^\.\n;]+)", cleaned, re.IGNORECASE)
        if match:
            candidate = _clean_birthplace(match.group(1))
            if candidate and _is_valid_birthplace_string(candidate):
                return candidate
    return None


def _extract_city_from_born_in(docs: list[Document]) -> Optional[str]:
    birthplace = _extract_birthplace_from_born_in(docs)
    if not birthplace:
        return None
    return _city_from_birthplace_string(birthplace)


def _clean_birthplace(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"^\s*born[:\s]+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^[A-Za-z]+\s+\d{1,2},\s+\d{4}\s*,?\s*", "", cleaned)
    cleaned = re.sub(r"^\d{4}\s*,?\s*", "", cleaned)
    cleaned = cleaned.split(" (born")[0]
    cleaned = cleaned.split(";")[0]
    cleaned = _strip_parentheticals(cleaned)
    cleaned = re.split(r"\b(?:is|was)\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = _strip_leading_in(cleaned)
    return cleaned.strip(" ,;])")


def _extract_birthplace_from_docs(docs: list[Document]) -> Optional[str]:
    patterns = [
        r"place of birth[:\s]+([^\n\.]+)",
        r"birthplace[:\s]+([^\n\.]+)",
        r"\bBorn\b\s*[:]?\s*([^\n\.]+)",
        r"\bwas born\b\s+in\s+([^\n\.]+)",
        r"\bborn\b[^.\n]{0,120}?,\s*([^\n\.]+)",
        r"\bborn\b[^.\n]{0,120}?\bin\s+([^\n\.]+)",
        r"\(born[^)]*\)\s*in\s+([^\n\.]+)",
    ]
    candidates: list[str] = []
    for doc in docs:
        text = doc.page_content or ""
        match = re.search(r"\bborn\b[^(\n]{0,80}?\(([^)]+)\)", text, re.IGNORECASE)
        if match:
            if not _contains_alias_marker(match.group(1)):
                candidates.append(match.group(1))
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                candidates.append(match.group(1))
    ranked: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = _clean_birthplace(candidate)
        if not cleaned:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        if not _is_valid_birthplace_string(cleaned):
            continue
        score = 0
        if "," in cleaned:
            score += 2
        if cleaned.count(",") >= 2:
            score += 1
        if re.search(r"\bU\.S\.|\bUSA\b|United States|United Kingdom|U\.K\.", cleaned, re.IGNORECASE):
            score += 2
        ranked.append((score, len(cleaned), cleaned))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][2]
    return None


def _city_from_birthplace_string(birthplace: str) -> Optional[str]:
    city = _strip_leading_in(birthplace.split(",")[0].strip())
    if city and _is_valid_city(city):
        return city
    return None


def _extract_city_from_docs(docs: list[Document]) -> Optional[str]:
    patterns = [
        r"place of birth[:\s]+([^\n\.]+)",
        r"birthplace[:\s]+([^\n\.]+)",
        r"\bborn\b[^.\n]{0,80}?\bin\s+([^\n\.]+)",
        r"\bborn\b[^(\n]{0,80}?\(([^)]+)\)",
    ]
    candidates: list[str] = []
    for doc in docs:
        text = doc.page_content or ""
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                candidates.append(match.group(1))
    for candidate in candidates:
        cleaned = _clean_birthplace(candidate)
        city = _city_from_birthplace_string(cleaned)
        if city:
            return city
    return None


def _inject_entity_passage_docs(
    table_links: list[str],
    passages: list[dict],
    docs: list[Document],
) -> list[Document]:
    if not table_links or not passages:
        return docs
    passages_by_url = {
        passage.get("url"): passage for passage in passages if passage.get("url")
    }
    existing_urls = {
        str((doc.metadata or {}).get("url") or "") for doc in docs if doc.metadata
    }
    forced_docs: list[Document] = []
    for link in table_links:
        if not link or link in existing_urls:
            continue
        passage = passages_by_url.get(link)
        if not passage:
            continue
        forced_docs.append(
            Document(
                page_content=passage.get("text", ""),
                metadata={
                    "source": "passage",
                    "url": link,
                    "title": passage.get("title", ""),
                    "forced": True,
                },
            )
        )
        existing_urls.add(link)
    if not forced_docs:
        return docs
    return forced_docs + docs


def _preferred_docs_for_links(docs: list[Document], table_links: list[str]) -> list[Document]:
    if not table_links:
        return []
    link_titles = {_title_from_link(link).lower() for link in table_links if link}
    link_urls = {link.lower() for link in table_links if link}
    preferred = []
    for doc in docs:
        meta = doc.metadata or {}
        url = str(meta.get("url") or "")
        title = str(meta.get("title") or "")
        if url and (url in table_links or url.lower() in link_urls):
            preferred.append(doc)
            continue
        doc_title = _title_from_link(url) if url else title
        if doc_title and doc_title.lower() in link_titles:
            preferred.append(doc)
    return preferred


def _extract_country_from_docs(docs: list[Document]) -> Optional[str]:
    us_states = {
        "Alabama",
        "Alaska",
        "Arizona",
        "Arkansas",
        "California",
        "Colorado",
        "Connecticut",
        "Delaware",
        "Florida",
        "Georgia",
        "Hawaii",
        "Idaho",
        "Illinois",
        "Indiana",
        "Iowa",
        "Kansas",
        "Kentucky",
        "Louisiana",
        "Maine",
        "Maryland",
        "Massachusetts",
        "Michigan",
        "Minnesota",
        "Mississippi",
        "Missouri",
        "Montana",
        "Nebraska",
        "Nevada",
        "New Hampshire",
        "New Jersey",
        "New Mexico",
        "New York",
        "North Carolina",
        "North Dakota",
        "Ohio",
        "Oklahoma",
        "Oregon",
        "Pennsylvania",
        "Rhode Island",
        "South Carolina",
        "South Dakota",
        "Tennessee",
        "Texas",
        "Utah",
        "Vermont",
        "Virginia",
        "Washington",
        "West Virginia",
        "Wisconsin",
        "Wyoming",
        "District of Columbia",
    }
    us_states_lower = {state.lower() for state in us_states}
    non_countries = {"state", "county", "province", "region", "district", "city"}
    patterns = [
        r"is a city in (?:the )?([A-Z][A-Za-z ]+)",
        r"city in (?:the )?([A-Z][A-Za-z ]+)",
        r"\bcountry[:\s]+([A-Z][A-Za-z ]+)\b",
        r"country of ([A-Z][A-Za-z ]+)",
        r"country is ([A-Z][A-Za-z ]+)",
        r"country in ([A-Z][A-Za-z ]+)",
        r"country[^\n]*?\b(?:in|of)\s+(?:the\s+)?([A-Z][A-Za-z ]+)",
    ]
    for doc in docs:
        text = doc.page_content
        if re.search(r"\bU\.S\.|\bUSA\b|United States", text, re.IGNORECASE):
            return "United States"
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                country = match.group(1).strip()
                if not country:
                    continue
                lowered = country.lower()
                if lowered in non_countries:
                    continue
                if lowered in us_states_lower:
                    continue
                return country
    return None


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _title_from_link(link: str) -> str:
    match = re.search(r"/wiki/([^#]+)", link)
    if not match:
        return link
    return match.group(1).replace("_", " ")


def _normalize_wiki_url(url: str) -> str:
    if not url:
        return ""
    if "/wiki/" in url:
        return "/wiki/" + url.split("/wiki/", 1)[1]
    return url


def find_passage_text_by_url(passages: list[dict], url: str) -> Optional[str]:
    target = _normalize_wiki_url(url)
    if not target:
        return None
    target_title = _title_from_link(target).lower()
    for passage in passages:
        passage_url = _normalize_wiki_url(str(passage.get("url") or ""))
        if passage_url and passage_url == target:
            return passage.get("text") or passage.get("content") or None
    for passage in passages:
        passage_url = str(passage.get("url") or "")
        title = passage.get("title") or _title_from_link(passage_url)
        if title and title.lower() == target_title:
            return passage.get("text") or passage.get("content") or None
    return None


def _build_rag_debug(state: AgentState) -> Optional[list[dict[str, object]]]:
    if not state.get("debug"):
        return None
    if state.get("rag_debug") is not None:
        return state.get("rag_debug")
    docs_by_hop = state.get("rag_docs_by_hop", {})
    rag_debug: list[dict[str, object]] = []
    for hop in sorted(docs_by_hop.keys()):
        docs = docs_by_hop[hop]
        doc_items = []
        for doc in docs:
            meta = doc.metadata or {}
            snippet = (doc.page_content or "")[:200]
            doc_items.append(
                {
                    "url": meta.get("url"),
                    "title": meta.get("title"),
                    "source": meta.get("source"),
                    "forced": bool(meta.get("forced")),
                    "preview": snippet,
                    "snippet": snippet,
                }
            )
        rag_debug.append({"hop": hop, "docs": doc_items})
    return rag_debug


def _attach_debug(payload: dict, state: AgentState) -> dict:
    rag_debug = _build_rag_debug(state)
    if rag_debug is not None:
        payload["rag_debug"] = rag_debug
    return payload


def load_data_node(state: AgentState) -> AgentState:
    tracker = ToolCallTracker()
    data_dir = Path(settings.data_dir)
    sample_dir = Path(settings.sample_dir)
    table_only = detect_table_only(state["question"])
    rag_k = state.get("rag_k")
    table_id, df = load_table(
        state.get("table_id"),
        state.get("table_path"),
        data_dir=data_dir,
        sample_dir=sample_dir,
        use_sample=state.get("use_sample", False),
    )
    table_schema = table_schema_text(df)
    function_calls = [
        {"name": "load_table", "arguments": {"table_id": table_id}},
    ]
    passages: list[dict] = []
    passages_loaded = False
    if not table_only and rag_k != 0:
        passages = load_passages(
            table_id=table_id,
            data_dir=data_dir,
            sample_dir=sample_dir,
            use_sample=state.get("use_sample", False),
        )
        passages_loaded = True
        function_calls.append({"name": "load_passages", "arguments": {"table_id": table_id}})
    return {
        "table_id": table_id,
        "df": df,
        "passages": passages,
        "passages_loaded": passages_loaded,
        "table_schema": table_schema,
        "tracker": tracker,
        "function_calls": function_calls,
        "table_only": table_only,
    }


def planner_node(state: AgentState) -> AgentState:
    if state.get("table_only"):
        return {"plan_steps": [], "need_rag": False}
    plan = plan_question(state["question"], state["table_schema"], settings)
    need_rag = (
        plan.get("need_rag", False)
        or _question_requires_rag(state["question"])
        or _needs_birthplace_city_country(state["question"])
    )
    return {
        "plan_steps": plan.get("steps", []),
        "need_rag": need_rag,
    }


def table_agent_node(state: AgentState) -> AgentState:
    if (
        _needs_country_only(state["question"])
        and _infer_row_index(state["question"]) is None
        and not state.get("table_only")
    ):
        return {
            "table_answer": "",
            "table_answer_structured": None,
            "table_reasoning": "Country-only question; table tools skipped.",
            "table_agent_mode": "router",
            "router_complete": False,
            "table_used_links": [],
            "need_rag": True,
        }
    router = route_table_question(state["question"], state["df"], state["tracker"])
    if router.get("handled"):
        requires_rag = _question_requires_rag(state["question"]) and not state.get("table_only")
        router_complete = bool(router.get("router_complete", True))
        if requires_rag:
            router_complete = False
        update = {
            "table_answer": router.get("answer_text", ""),
            "table_answer_structured": router.get("answer_structured"),
            "table_reasoning": router.get("reasoning", ""),
            "table_agent_mode": "router",
            "table_used_links": router.get("used_links", []),
            "router_complete": router_complete,
        }
        if requires_rag:
            update["need_rag"] = True
        if router.get("disable_rag") and not (
            _needs_birthplace_country(state["question"])
            or _needs_birthplace_city_country(state["question"])
            or _question_requires_rag(state["question"])
        ):
            update["need_rag"] = False
            update["rag_k"] = 0
        return update
    table_answer = run_table_agent(
        question=state["question"],
        table_schema=state["table_schema"],
        df=state["df"],
        settings=settings,
        tracker=state["tracker"],
        plan_steps=state.get("plan_steps"),
    )
    return {
        "table_answer": table_answer,
        "table_answer_structured": None,
        "table_reasoning": "",
        "table_agent_mode": "llm",
        "router_complete": False,
        "table_used_links": [],
    }


def rag_node(state: AgentState) -> AgentState:
    rag_k = state.get("rag_k")
    force_multihop = (
        _needs_birthplace_country(state["question"])
        or _needs_birthplace_city_country(state["question"])
    )

    if state.get("table_only"):
        sources = list(state.get("table_used_links", []))
        return {
            "rag_docs": [],
            "rag_context": [],
            "sources": sources,
            "rag_debug": [] if state.get("debug") else None,
        }

    if rag_k == 0 and not force_multihop:
        sources = list(state.get("table_used_links", []))
        return {
            "rag_docs": [],
            "rag_context": [],
            "sources": sources,
            "rag_debug": [] if state.get("debug") else None,
        }
    if rag_k == 0 and force_multihop:
        rag_k = None
    need_rag = (
        state.get("need_rag", False)
        or _question_requires_rag(state["question"])
        or _needs_birthplace_city_country(state["question"])
        or rag_k is not None
    )
    if not need_rag:
        sources = list(state.get("table_used_links", []))
        return {
            "rag_docs": [],
            "rag_context": [],
            "sources": sources,
            "rag_debug": [] if state.get("debug") else None,
        }
    k = rag_k if rag_k is not None else settings.max_rag_results
    question = state["question"]
    if _needs_birthplace_country(question) and settings.rag_chunk_size < 800:
        settings.rag_chunk_size = 800
    tracker = state["tracker"]
    function_calls = list(state.get("function_calls", []))
    passages = state.get("passages", [])
    passages_loaded = state.get("passages_loaded", False)
    if not passages_loaded:
        passages = load_passages(
            table_id=state["table_id"],
            data_dir=Path(settings.data_dir),
            sample_dir=Path(settings.sample_dir),
            use_sample=state.get("use_sample", False),
        )
        passages_loaded = True
        function_calls.append({"name": "load_passages", "arguments": {"table_id": state["table_id"]}})
    row_index = _infer_row_index(question)
    table_links: list[str] = []
    links_from_cell: list[str] = []
    entity: Optional[str] = None
    question_entity: Optional[str] = None
    if row_index is None and _needs_country_only(question):
        question_entity = _extract_location_from_question(question)
    if row_index is None and not question_entity:
        question_entity = _extract_named_entity_from_question(question)
    if row_index is not None:
        entity, links, _column = _select_entity_from_row(state["df"], row_index, question)
        if links:
            links_from_cell.extend([link for link in links if link])
    table_links = _dedupe(
        [link for link in state.get("table_used_links", []) if link] + links_from_cell
    )
    anchor_title = ""
    if table_links:
        anchor_title = _title_from_link(table_links[0])
        base_query = anchor_title
    elif question_entity:
        base_query = question_entity
    else:
        base_query = entity or ""
    if not base_query or _is_numeric_string(base_query):
        base_query = question
    lowered = question.lower()
    if any(token in lowered for token in ("birthplace", "birth place", "place of birth", "born", "место рождения", "родился")):
        hop_query = f"{base_query} place of birth born in"
    elif any(
        token in lowered
        for token in ("country", "located in", "what country", "which country", "стране", "страна")
    ):
        hop_query = f"{base_query} country"
    else:
        hop_query = base_query if base_query == question else f"{base_query} {question}"
    k_search = max(k, min(k * 2, 20))
    table_texts = []
    for idx, row in state["df"].iterrows():
        parts = [f"{col}={row[col]}" for col in state["df"].columns]
        table_texts.append(
            {
                "text": f"row_{idx}: " + ", ".join(parts),
                "metadata": {"source_type": "row", "row_index": int(idx)},
            }
        )
    for col in state["df"].columns:
        table_texts.append(
            {
                "text": f"column:{col}",
                "metadata": {"source_type": "column", "column_name": str(col)},
            }
        )
    docs_all: list[Document] = []
    docs_by_hop: dict[int, list[Document]] = {}
    rag_hops: list[dict] = []
    rag_debug: list[dict[str, object]] = list(state.get("rag_debug", []))
    force_multihop = _needs_birthplace_country(question) or _needs_birthplace_city_country(question)
    multihop_enabled = settings.rag_enable_multihop or force_multihop
    max_hops = settings.rag_max_hops if multihop_enabled else 1
    if force_multihop:
        max_hops = max(max_hops, 2)

    for hop in range(max_hops):
        tracker.record("rag_search", {"query": hop_query, "k": k, "hop": hop})
        hop_k = k_search if hop == 0 else k
        if hop == 0 and _needs_birthplace_country(question):
            hop_k = max(hop_k, 10)
        docs_raw = rag_search(
            table_id=state["table_id"],
            table_texts=table_texts,
            passages=passages,
            settings=settings,
            query=hop_query,
            k=hop_k,
        )
        if hop == 0:
            docs_raw = _inject_entity_passage_docs(table_links, passages, docs_raw)
        passage_docs = [
            doc for doc in docs_raw if (doc.metadata or {}).get("source") == "passage"
        ]
        table_docs = [
            doc for doc in docs_raw if (doc.metadata or {}).get("source") != "passage"
        ]
        docs_ranked = passage_docs + table_docs
        if state.get("debug"):
            debug_docs = []
            for doc in docs_ranked[:k]:
                meta = doc.metadata or {}
                snippet = (doc.page_content or "")[:200]
                debug_docs.append(
                    {
                        "title": meta.get("title"),
                        "url": meta.get("url"),
                        "snippet": snippet,
                        "preview": snippet,
                    }
                )
            rag_debug.append({"hop": hop, "query": hop_query, "docs": debug_docs})
        fallback_docs_count = 0
        fallback_query = ""
        birthplace = None
        if _needs_birthplace_country(question) or _needs_birthplace_city_country(question):
            preferred_docs = _preferred_docs_for_links(docs_ranked, table_links)
            docs_for_birthplace = preferred_docs or docs_ranked
            birthplace = _extract_birthplace_from_docs(docs_for_birthplace)
            if not birthplace and docs_ranked:
                birthplace = _extract_birthplace_from_docs([docs_ranked[0]])
            born_in_candidate = None
            if _needs_birthplace_city_country(state["question"]):
                born_in_candidate = _extract_birthplace_from_born_in(docs_ranked)
                if born_in_candidate:
                    if not birthplace:
                        birthplace = born_in_candidate
                    elif "," not in birthplace and "," in born_in_candidate:
                        birthplace = born_in_candidate
            if not birthplace and table_links:
                passage_text = find_passage_text_by_url(passages, table_links[0])
                if passage_text:
                    tracker.record("passage_lookup", {"url": table_links[0]})
                    passage_doc = Document(
                        page_content=passage_text,
                        metadata={
                            "source": "passage",
                            "url": table_links[0],
                            "title": _title_from_link(table_links[0]),
                            "forced": True,
                        },
                    )
                    docs_ranked = [passage_doc] + docs_ranked
                    birthplace = _extract_birthplace_from_docs([passage_doc])
                    if not birthplace:
                        birthplace = _extract_birthplace_from_docs(docs_ranked)
            if not birthplace and anchor_title:
                fallback_queries = [
                    f"{anchor_title} Born:",
                    f"{anchor_title} place of birth",
                    f"{anchor_title} birthplace",
                ]
                fallback_k = max(hop_k, 10)
                for fallback_query in fallback_queries:
                    tracker.record(
                        "rag_search",
                        {"query": fallback_query, "k": fallback_k, "hop": hop, "fallback": "birthplace"},
                    )
                    fallback_raw = rag_search(
                        table_id=state["table_id"],
                        table_texts=table_texts,
                        passages=passages,
                        settings=settings,
                        query=fallback_query,
                        k=fallback_k,
                    )
                    fallback_raw = _inject_entity_passage_docs(table_links, passages, fallback_raw)
                    fallback_passages = [
                        doc for doc in fallback_raw if (doc.metadata or {}).get("source") == "passage"
                    ]
                    fallback_table = [
                        doc for doc in fallback_raw if (doc.metadata or {}).get("source") != "passage"
                    ]
                    fallback_ranked = fallback_passages + fallback_table
                    fallback_docs_count = len(fallback_ranked)
                    if state.get("debug"):
                        debug_docs = []
                        for doc in fallback_ranked[:k]:
                            meta = doc.metadata or {}
                            snippet = (doc.page_content or "")[:200]
                            debug_docs.append(
                                {
                                    "title": meta.get("title"),
                                    "url": meta.get("url"),
                                    "snippet": snippet,
                                    "preview": snippet,
                                }
                            )
                        rag_debug.append(
                            {
                                "hop": hop,
                                "query": fallback_query,
                                "docs": debug_docs,
                                "fallback": True,
                            }
                        )
                    docs_ranked = fallback_ranked + docs_ranked
                    preferred_docs = _preferred_docs_for_links(docs_ranked, table_links)
                    docs_for_birthplace = preferred_docs or docs_ranked
                    birthplace = _extract_birthplace_from_docs(docs_for_birthplace)
                    if birthplace:
                        break
            if _needs_birthplace_city_country(state["question"]) and docs_ranked:
                born_in_candidate = _extract_birthplace_from_born_in(docs_ranked)
                if born_in_candidate and (not birthplace or "," not in birthplace):
                    birthplace = born_in_candidate
            if birthplace and not _is_valid_birthplace_string(birthplace):
                birthplace = None
        docs_top = docs_ranked[:k]
        docs_all.extend(docs_top)
        docs_by_hop[hop] = docs_top
        rag_hops.append({"hop": hop, "query": hop_query, "docs": len(docs_top)})
        if fallback_docs_count:
            rag_hops.append(
                {
                    "hop": hop,
                    "query": fallback_query,
                    "docs": fallback_docs_count,
                    "fallback": True,
                }
            )
        if not multihop_enabled or hop >= max_hops - 1:
            break

        if (_needs_birthplace_country(question) or _needs_birthplace_city_country(question)) and not birthplace:
            city = _extract_city_from_docs(docs_ranked)
            if city:
                tracker.record("extract_intermediate", {"type": "city_from_docs", "city": city, "hop": hop})
                hop_query = f"{city} country"
                continue

        if (_needs_birthplace_country(state["question"]) or _needs_birthplace_city_country(state["question"])) and birthplace:
            clean_birthplace = _strip_leading_in(birthplace)
            tracker.record("extract_intermediate", {"type": "birthplace", "birthplace": clean_birthplace, "hop": hop})
            city = _city_from_birthplace_string(clean_birthplace)
            if city:
                tracker.record("extract_intermediate", {"type": "city_from_birthplace", "city": city, "hop": hop})
                hop_query = f"{city} country"
                continue
            hop_query = f"{clean_birthplace} country"
            continue
        break

    limit = max(k, settings.max_rag_results) * max(1, len(rag_hops))
    rag_context = [doc.page_content for doc in docs_all][:limit]
    sources_raw = list(state.get("table_used_links", [])) + table_links
    sources = _dedupe([source for source in sources_raw if source])
    seen = set(sources)
    for doc in docs_all:
        meta = doc.metadata or {}
        source = meta.get("url") or meta.get("title") or meta.get("source")
        if source and source not in seen:
            seen.add(source)
            sources.append(str(source))
    return {
        "rag_docs": docs_all,
        "rag_docs_by_hop": docs_by_hop,
        "rag_context": rag_context,
        "sources": sources,
        "passages": passages,
        "passages_loaded": passages_loaded,
        "function_calls": function_calls,
        "rag_hops": rag_hops,
        "table_links": table_links,
        "question_entity": question_entity,
        "rag_debug": rag_debug if state.get("debug") else None,
    }


def analysis_node(state: AgentState) -> AgentState:
    table_answer = state.get("table_answer", "")
    table_structured = state.get("table_answer_structured")
    table_reasoning = state.get("table_reasoning", "")
    if state.get("table_only") and not state.get("rag_context"):
        final_answer = table_answer or "No answer produced."
        reasoning = table_reasoning or "Table-only; analysis skipped."
        function_calls = list(state.get("function_calls", []))
        function_calls.extend(state["tracker"].calls)
        return _attach_debug(
            {
            "final_answer": final_answer,
            "final_answer_structured": table_structured,
            "reasoning": reasoning,
            "function_calls": function_calls,
            "sources": state.get("sources", []),
            },
            state,
        )
    analysis = None
    if state.get("table_answer") == "insufficient_tool_evidence":
        analysis = {
            "final_answer": "insufficient_tool_evidence",
            "reasoning": "No table tools were called; answer is not grounded.",
            "error": "insufficient_tool_evidence",
        }
    if analysis is None and _needs_birthplace_country(state["question"]):
        docs_by_hop = state.get("rag_docs_by_hop", {})
        hop0_docs = docs_by_hop.get(0, [])
        hop1_docs = docs_by_hop.get(1, [])
        table_links = state.get("table_links", [])
        preferred_docs = _preferred_docs_for_links(hop0_docs, table_links)
        docs_for_birthplace = preferred_docs or hop0_docs
        birthplace = _extract_birthplace_from_docs(docs_for_birthplace) if hop0_docs else None
        if birthplace and not _is_valid_birthplace_string(birthplace):
            birthplace = None
        country = _extract_country_from_docs(hop1_docs) if hop1_docs else None
        if not country and birthplace:
            if re.search(r"\bU\.S\.|\bUSA\b|United States", birthplace, re.IGNORECASE):
                country = "United States"
        if birthplace and country:
            final_answer = f"{birthplace} — {country}"
            reasoning = (
                "From retrieved passages: extracted birthplace from the entity page, then "
                "queried the city to identify its country."
            )
            function_calls = list(state.get("function_calls", []))
            function_calls.extend(state["tracker"].calls)
            return _attach_debug(
                {
                "final_answer": final_answer,
                "final_answer_structured": {
                    "birthplace": birthplace,
                    "country": country,
                },
                "reasoning": reasoning,
                "function_calls": function_calls,
                "sources": state.get("sources", []),
                },
                state,
            )
        if birthplace:
            final_answer = birthplace
            reasoning = (
                "From retrieved passages: extracted birthplace; "
                "country not resolved due to missing evidence."
            )
            function_calls = list(state.get("function_calls", []))
            function_calls.extend(state["tracker"].calls)
            return _attach_debug(
                {
                "final_answer": final_answer,
                "final_answer_structured": {"birthplace": birthplace, "country": None},
                "reasoning": reasoning,
                "function_calls": function_calls,
                "sources": state.get("sources", []),
                },
                state,
            )
        reasoning = "Birthplace not found in retrieved passages."
        if not state.get("rag_context"):
            reasoning = "No passages retrieved or birthplace not found in passages."
        function_calls = list(state.get("function_calls", []))
        function_calls.extend(state["tracker"].calls)
        return _attach_debug(
            {
                "final_answer": "Unknown",
            "final_answer_structured": {
                "birthplace": None,
                "country": None,
                "error": "not_found_in_passages",
            },
            "reasoning": reasoning,
            "function_calls": function_calls,
            "sources": state.get("sources", []),
            },
            state,
        )
    if analysis is None and _needs_country_only(state["question"]):
        docs_by_hop = state.get("rag_docs_by_hop", {})
        hop0_docs = docs_by_hop.get(0, [])
        question_entity = state.get("question_entity") or _extract_location_from_question(state["question"])
        table_links = state.get("table_links", [])
        if (
            question_entity
            and not table_links
            and hop0_docs
            and not _entity_mentioned_in_docs(question_entity, hop0_docs)
        ):
            function_calls = list(state.get("function_calls", []))
            function_calls.extend(state["tracker"].calls)
            return _attach_debug(
                {
                    "final_answer": "Unknown",
                    "final_answer_structured": {"country": None, "error": "entity_not_in_passages"},
                    "reasoning": "Requested location was not present in retrieved passages.",
                    "function_calls": function_calls,
                    "sources": state.get("sources", []),
                },
                state,
            )
        country = _extract_country_from_docs(hop0_docs) if hop0_docs else None
        if country:
            structured = {"country": country} if "json" in state["question"].lower() else None
            reasoning = (
                "From retrieved passages: extracted country for the linked location."
            )
            function_calls = list(state.get("function_calls", []))
            function_calls.extend(state["tracker"].calls)
            return _attach_debug(
                {
                    "final_answer": country if not structured else json.dumps(structured, ensure_ascii=True),
                    "final_answer_structured": structured,
                    "reasoning": reasoning,
                    "function_calls": function_calls,
                    "sources": state.get("sources", []),
                },
                state,
            )
        function_calls = list(state.get("function_calls", []))
        function_calls.extend(state["tracker"].calls)
        return _attach_debug(
            {
                "final_answer": "Unknown",
                "final_answer_structured": {"country": None, "error": "not_found_in_passages"},
                "reasoning": "Country not found in retrieved passages.",
                "function_calls": function_calls,
                "sources": state.get("sources", []),
            },
            state,
        )
    if (
        analysis is None
        and state.get("table_agent_mode") == "router"
        and not state.get("rag_context")
        and state.get("router_complete")
    ):
        final_answer = table_answer or "No answer produced."
        reasoning = table_reasoning or "Router answer used without RAG context."
        function_calls = list(state.get("function_calls", []))
        function_calls.extend(state["tracker"].calls)
        return _attach_debug(
            {
            "final_answer": final_answer,
            "final_answer_structured": table_structured,
            "reasoning": reasoning,
            "function_calls": function_calls,
            "sources": state.get("sources", []),
            },
            state,
        )
    if analysis is None:
        analysis = run_analysis_agent(
            question=state["question"],
            table_answer=table_answer,
            rag_context=state.get("rag_context", []),
            settings=settings,
            plan_steps=state.get("plan_steps"),
        )
    final_answer = analysis.get("final_answer", "")
    reasoning = analysis.get("reasoning", "")
    final_answer_structured = analysis.get("final_answer_structured")
    if analysis.get("error") or not final_answer or not reasoning:
        final_answer = table_answer or "No answer produced."
        if _needs_birthplace_country(state["question"]) and state.get("rag_context"):
            reasoning = "Fallback: used RAG context for birthplace/country."
        else:
            reasoning = (
                table_reasoning
                or "Fallback: analysis agent returned an invalid or empty response."
            )
        final_answer_structured = table_structured
    if not isinstance(final_answer, str):
        final_answer_structured = final_answer
        final_answer = json.dumps(final_answer, ensure_ascii=True)
    if isinstance(final_answer, str):
        stripped = final_answer.strip()
        if stripped and (stripped[0] in ("{", "[") or re.match(r"^-?\d+(\.\d+)?$", stripped)):
            try:
                parsed = extract_json(stripped)
                final_answer_structured = parsed
            except Exception:
                pass
    rag_used = bool(state.get("rag_context")) or any(
        "/wiki/" in str(source) for source in state.get("sources", [])
    )
    if rag_used and isinstance(reasoning, str):
        reasoning = re.sub(
            r"(?i)table (indicates|provides|shows|showed)",
            "retrieved passages indicate",
            reasoning,
        )
        reasoning = re.sub(
            r"(?i)table (show|shows|showed)",
            "retrieved passages show",
            reasoning,
        )
        reasoning = re.sub(
            r"(?i)from the table",
            "From the table we got the linked entity; from retrieved passages",
            reasoning,
        )
    if not isinstance(reasoning, str):
        reasoning = json.dumps(reasoning, ensure_ascii=True)
    function_calls = list(state.get("function_calls", []))
    function_calls.extend(state["tracker"].calls)
    return _attach_debug(
        {
        "final_answer": final_answer,
        "final_answer_structured": final_answer_structured,
        "reasoning": reasoning,
        "function_calls": function_calls,
        "sources": state.get("sources", []),
        },
        state,
    )


@lru_cache(maxsize=1)
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("load_data", load_data_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("table_agent", table_agent_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("analysis", analysis_node)

    workflow.set_entry_point("load_data")
    workflow.add_edge("load_data", "planner")
    workflow.add_edge("planner", "table_agent")
    workflow.add_edge("table_agent", "rag")
    workflow.add_edge("rag", "analysis")
    workflow.add_edge("analysis", END)

    return workflow.compile()


def run_agent(
    question: str,
    table_id: Optional[str],
    table_path: Optional[str],
    use_sample: bool,
    rag_k: Optional[int],
    debug: bool,
):
    graph = build_graph()
    state = {
        "question": question,
        "table_id": table_id,
        "table_path": table_path,
        "use_sample": use_sample,
        "rag_k": rag_k,
        "debug": debug,
    }
    return graph.invoke(state)
