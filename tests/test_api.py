from __future__ import annotations

import json
import re
from typing import Any
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import app.agents.graph as graph
from app.agents.tools import ToolCallTracker, build_table_tools
from app.rag import loader
from app.main import app


@pytest.fixture(autouse=True)
def stub_data(monkeypatch):
    df = pd.DataFrame(
        [
            {"Player": "Alice", "Name": "Art Museum"},
            {"Player": "Bob", "Name": "History Museum"},
        ]
    )
    df.attrs["cell_links"] = {(0, "Player"): ["/wiki/Alice"]}
    df.attrs["header_links"] = {}

    def _load_table(*_args: Any, **_kwargs: Any):
        return "test_table", df

    def _load_passages(*_args: Any, **_kwargs: Any):
        return []

    def _plan_question(*_args: Any, **_kwargs: Any):
        return {"steps": ["stub"], "need_rag": True, "notes": "stub"}

    def _rag_search(*_args: Any, **_kwargs: Any):
        return []

    def _run_analysis_agent(*_args: Any, **_kwargs: Any):
        return {"final_answer": "ok", "reasoning": "stub"}

    monkeypatch.setattr(graph, "load_table", _load_table)
    monkeypatch.setattr(graph, "load_passages", _load_passages)
    monkeypatch.setattr(graph, "plan_question", _plan_question)
    monkeypatch.setattr(graph, "rag_search", _rag_search)
    monkeypatch.setattr(graph, "run_analysis_agent", _run_analysis_agent)
    return df


def test_json_with_quotes():
    client = TestClient(app)
    payload = {
        "question": 'Find rows where the museum name contains "Art"',
        "table_id": "test_table",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == payload["question"]


def test_tool_use_first_row():
    client = TestClient(app)
    payload = {
        "question": "What is the player name in the first row of the table?",
        "table_id": "test_table",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    names = [call["name"] for call in response.json()["payload"]["function_calls"]]
    assert "get_row" in names or "lookup_cell" in names


def test_contains_uses_filter_tool():
    client = TestClient(app)
    payload = {
        "question": "Find rows where the museum name contains 'Art'",
        "table_id": "test_table",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    names = [call["name"] for call in response.json()["payload"]["function_calls"]]
    assert "filter_rows_contains" in names


def test_rag_k_zero_disables_rag():
    client = TestClient(app)
    payload = {
        "question": "What is the player name in the first row of the table?",
        "table_id": "test_table",
        "rag_k": 0,
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    names = [call["name"] for call in response.json()["payload"]["function_calls"]]
    assert "rag_search" not in names


def test_path_traversal_blocked():
    client = TestClient(app)
    payload = {
        "question": "Test",
        "table_id": "../etc/passwd",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 400


def test_sample_data_present():
    assert Path("data/sample/tables/sample_table.csv").exists()
    assert Path("data/sample/passages/sample_passages.json").exists()


def test_columns_router_table_only():
    client = TestClient(app)
    payload = {
        "question": 'Return the column names as JSON: {"columns": [...]} Use only the table.',
        "table_id": "test_table",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    names = [call["name"] for call in data["payload"]["function_calls"]]
    assert "get_columns" in names
    assert "rag_search" not in names
    assert data["payload"]["final_answer_structured"] == {"columns": ["Player", "Name"]}


def test_links_json_router():
    client = TestClient(app)
    payload = {
        "question": 'Return wiki links for row 0 under column "Player" as JSON: {"links": [...]} Use only the table.',
        "table_id": "test_table",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    names = [call["name"] for call in data["payload"]["function_calls"]]
    assert "lookup_links" in names
    assert data["payload"]["final_answer_structured"] == {"links": ["/wiki/Alice"]}


def test_links_parsed_from_tables_tok(tmp_path):
    payload = {
        "header": [["Name", ["/wiki/Header"]]],
        "data": [
            [["Alice", ["/wiki/Alice"]]],
        ],
    }
    tables_dir = tmp_path / "tables_tok"
    tables_dir.mkdir(parents=True)
    table_path = tables_dir / "sample.json"
    table_path.write_text(json.dumps(payload), encoding="utf-8")
    table_id, df = loader.load_table(
        table_id=None,
        table_path=str(table_path),
        data_dir=tmp_path,
        sample_dir=tmp_path,
        use_sample=False,
    )
    assert table_id == "sample"
    assert df.attrs["cell_links"][(0, "Name")] == ["/wiki/Alice"]
    assert df.attrs["header_links"]["Name"] == ["/wiki/Header"]
    tracker = ToolCallTracker()
    tools = {tool.name: tool for tool in build_table_tools(df, tracker)}
    links = tools["lookup_links"].invoke({"row_index": 0, "column": "Name"})
    assert links == ["/wiki/Alice"]
    assert any(call["name"] == "lookup_links" for call in tracker.calls)


def test_multihop_rag_calls_twice_for_birthplace_country(monkeypatch):
    df = pd.DataFrame([{"Person": "Alice"}])
    df.attrs["cell_links"] = {(0, "Person"): ["/wiki/Alice"]}
    df.attrs["header_links"] = {}

    def _load_table(*_args: Any, **_kwargs: Any):
        return "test_table", df

    def _load_passages(*_args: Any, **_kwargs: Any):
        return []

    def _plan_question(*_args: Any, **_kwargs: Any):
        return {"steps": ["stub"], "need_rag": True, "notes": "stub"}

    def _run_analysis_agent(*_args: Any, **_kwargs: Any):
        return {"final_answer": "France", "reasoning": "stub"}

    def _rag_search(*_args: Any, **kwargs: Any):
        query = kwargs.get("query", "")
        if "country" in query:
            return [
                Document(
                    page_content="Pensacola is a city in the United States.",
                    metadata={"source": "passage", "url": "/wiki/Pensacola"},
                )
            ]
        return [
            Document(
                page_content="Born: June 1, 1900 (Pensacola, Florida, U.S.)",
                metadata={"source": "passage", "url": "/wiki/Alice"},
            )
        ]

    monkeypatch.setattr(graph, "load_table", _load_table)
    monkeypatch.setattr(graph, "load_passages", _load_passages)
    monkeypatch.setattr(graph, "plan_question", _plan_question)
    monkeypatch.setattr(graph, "run_analysis_agent", _run_analysis_agent)
    monkeypatch.setattr(graph, "rag_search", _rag_search)

    client = TestClient(app)
    payload = {
        "question": "What is the birth place of the first person in the table, and which country is that city in?",
        "table_id": "test_table",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    calls = response.json()["payload"]["function_calls"]
    rag_calls = [call for call in calls if call["name"] == "rag_search"]
    hops = {call["arguments"].get("hop") for call in rag_calls}
    assert 0 in hops and 1 in hops
    assert "United States" in response.json()["payload"]["final_answer"]


def test_multihop_city_validation_rejects_nfl(monkeypatch):
    df = pd.DataFrame([{"Player": "Emmitt Smith"}])
    df.attrs["cell_links"] = {(0, "Player"): ["/wiki/Emmitt_Smith"]}
    df.attrs["header_links"] = {}

    def _load_table(*_args: Any, **_kwargs: Any):
        return "test_table", df

    def _load_passages(*_args: Any, **_kwargs: Any):
        return []

    def _plan_question(*_args: Any, **_kwargs: Any):
        return {"steps": ["stub"], "need_rag": True, "notes": "stub"}

    def _rag_search(*_args: Any, **kwargs: Any):
        query = kwargs.get("query", "")
        if "country" in query:
            return [
                Document(
                    page_content="Pensacola is a city in the United States.",
                    metadata={"source": "passage", "url": "/wiki/Pensacola"},
                )
            ]
        return [
            Document(
                page_content="born in NFL season",
                metadata={"source": "passage", "url": "/wiki/NFL"},
            ),
            Document(
                page_content="born in Pensacola, Florida.",
                metadata={"source": "passage", "url": "/wiki/Emmitt_Smith"},
            ),
        ]

    monkeypatch.setattr(graph, "load_table", _load_table)
    monkeypatch.setattr(graph, "load_passages", _load_passages)
    monkeypatch.setattr(graph, "plan_question", _plan_question)
    monkeypatch.setattr(graph, "rag_search", _rag_search)

    client = TestClient(app)
    payload = {
        "question": "What is the birthplace of the first player in the table, and which country is that city in?",
        "table_id": "test_table",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    calls = response.json()["payload"]["function_calls"]
    extract_calls = [call for call in calls if call["name"] == "extract_intermediate"]
    city_calls = [
        call
        for call in extract_calls
        if call["arguments"].get("type") == "city_from_birthplace"
    ]
    assert city_calls
    assert city_calls[0]["arguments"]["city"] == "Pensacola"
    rag_calls = [
        call for call in calls if call["name"] == "rag_search" and call["arguments"].get("hop") == 1
    ]
    assert any("Pensacola" in call["arguments"].get("query", "") for call in rag_calls)


def test_multihop_country_extraction(monkeypatch):
    df = pd.DataFrame([{"Player": "Emmitt Smith"}])
    df.attrs["cell_links"] = {(0, "Player"): ["/wiki/Emmitt_Smith"]}
    df.attrs["header_links"] = {}

    def _load_table(*_args: Any, **_kwargs: Any):
        return "test_table", df

    def _load_passages(*_args: Any, **_kwargs: Any):
        return []

    def _plan_question(*_args: Any, **_kwargs: Any):
        return {"steps": ["stub"], "need_rag": True, "notes": "stub"}

    def _rag_search(*_args: Any, **kwargs: Any):
        query = kwargs.get("query", "")
        if "country" in query:
            return [
                Document(
                    page_content="Pensacola is a city in the United States.",
                    metadata={"source": "passage", "url": "/wiki/Pensacola"},
                )
            ]
        return [
            Document(
                page_content="place of birth: Pensacola, Florida.",
                metadata={"source": "passage", "url": "/wiki/Emmitt_Smith"},
            )
        ]

    monkeypatch.setattr(graph, "load_table", _load_table)
    monkeypatch.setattr(graph, "load_passages", _load_passages)
    monkeypatch.setattr(graph, "plan_question", _plan_question)
    monkeypatch.setattr(graph, "rag_search", _rag_search)

    client = TestClient(app)
    payload = {
        "question": "What is the birthplace of the first player in the table, and which country is that city in?",
        "table_id": "test_table",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    assert "United States" in response.json()["payload"]["final_answer"]


def test_multihop_birthplace_country_uses_entity_passage(monkeypatch):
    df = pd.DataFrame([{"Player": "Emmitt Smith"}])
    df.attrs["cell_links"] = {(0, "Player"): ["/wiki/Emmitt_Smith"]}
    df.attrs["header_links"] = {}

    def _load_table(*_args: Any, **_kwargs: Any):
        return "test_table", df

    def _load_passages(*_args: Any, **_kwargs: Any):
        return [
            {
                "url": "/wiki/Emmitt_Smith",
                "title": "Emmitt Smith",
                "text": "Emmitt Smith was born in Pensacola, Florida.",
            }
        ]

    def _plan_question(*_args: Any, **_kwargs: Any):
        return {"steps": ["stub"], "need_rag": True, "notes": "stub"}

    def _rag_search(*_args: Any, **kwargs: Any):
        query = kwargs.get("query", "")
        if "country" in query:
            return [
                Document(
                    page_content="Pensacola is a city in the United States.",
                    metadata={"source": "passage", "url": "/wiki/Pensacola"},
                )
            ]
        return [
            Document(
                page_content="born in NFL season",
                metadata={"source": "passage", "url": "/wiki/NFL"},
            )
        ]

    monkeypatch.setattr(graph, "load_table", _load_table)
    monkeypatch.setattr(graph, "load_passages", _load_passages)
    monkeypatch.setattr(graph, "plan_question", _plan_question)
    monkeypatch.setattr(graph, "rag_search", _rag_search)

    client = TestClient(app)
    payload = {
        "question": "What is the birthplace of the first player in the table, and which country is that city in?",
        "table_id": "test_table",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    calls = response.json()["payload"]["function_calls"]
    rag_calls = [call for call in calls if call["name"] == "rag_search"]
    assert {call["arguments"].get("hop") for call in rag_calls} >= {0, 1}
    assert "United States" in response.json()["payload"]["final_answer"]


def test_router_does_not_short_circuit_multihop_birthplace_country(monkeypatch):
    df = pd.DataFrame([{"Player": "Emmitt Smith"}])
    df.attrs["cell_links"] = {(0, "Player"): ["/wiki/Emmitt_Smith"]}
    df.attrs["header_links"] = {}

    def _load_table(*_args: Any, **_kwargs: Any):
        return "test_table", df

    def _load_passages(*_args: Any, **_kwargs: Any):
        return [
            {
                "url": "/wiki/Emmitt_Smith",
                "title": "Emmitt Smith",
                "text": "Emmitt Smith was born in Pensacola, Florida.",
            }
        ]

    def _plan_question(*_args: Any, **_kwargs: Any):
        return {"steps": ["stub"], "need_rag": False, "notes": "stub"}

    def _rag_search(*_args: Any, **kwargs: Any):
        query = kwargs.get("query", "")
        if "country" in query:
            return [
                Document(
                    page_content="Pensacola is a city in the United States.",
                    metadata={"source": "passage", "url": "/wiki/Pensacola"},
                )
            ]
        return [
            Document(
                page_content="place of birth: Pensacola, Florida.",
                metadata={"source": "passage", "url": "/wiki/Emmitt_Smith"},
            )
        ]

    monkeypatch.setattr(graph, "load_table", _load_table)
    monkeypatch.setattr(graph, "load_passages", _load_passages)
    monkeypatch.setattr(graph, "plan_question", _plan_question)
    monkeypatch.setattr(graph, "rag_search", _rag_search)

    client = TestClient(app)
    payload = {
        "question": "What is the birthplace of the first player in the table, and which country is that city in?",
        "table_id": "test_table",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    calls = response.json()["payload"]["function_calls"]
    assert any(call["name"] == "rag_search" for call in calls)


def test_router_does_not_short_circuit_country_located(monkeypatch):
    df = pd.DataFrame([{"Name": "APEX Museum", "City": "Atlanta"}])
    df.attrs["cell_links"] = {(0, "City"): ["/wiki/Atlanta"]}
    df.attrs["header_links"] = {}

    def _load_table(*_args: Any, **_kwargs: Any):
        return "test_table", df

    def _load_passages(*_args: Any, **_kwargs: Any):
        return [
            {
                "url": "/wiki/Atlanta",
                "title": "Atlanta",
                "text": "Atlanta is a city in the United States.",
            }
        ]

    def _plan_question(*_args: Any, **_kwargs: Any):
        return {"steps": ["stub"], "need_rag": False, "notes": "stub"}

    def _rag_search(*_args: Any, **kwargs: Any):
        return [
            Document(
                page_content="Atlanta is a city in the United States.",
                metadata={"source": "passage", "url": "/wiki/Atlanta"},
            )
        ]

    monkeypatch.setattr(graph, "load_table", _load_table)
    monkeypatch.setattr(graph, "load_passages", _load_passages)
    monkeypatch.setattr(graph, "plan_question", _plan_question)
    monkeypatch.setattr(graph, "rag_search", _rag_search)

    client = TestClient(app)
    payload = {
        "question": "What country is the city linked in the first row located in?",
        "table_id": "test_table",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    calls = response.json()["payload"]["function_calls"]
    assert any(call["name"] == "rag_search" for call in calls)


def test_rag_query_not_numeric_for_birthplace(monkeypatch):
    df = pd.DataFrame([{"Rank": 1, "Player": "Emmitt Smith"}])
    df.attrs["cell_links"] = {(0, "Player"): ["/wiki/Emmitt_Smith"]}
    df.attrs["header_links"] = {}
    seen_queries: list[str] = []

    def _load_table(*_args: Any, **_kwargs: Any):
        return "test_table", df

    def _load_passages(*_args: Any, **_kwargs: Any):
        return []

    def _plan_question(*_args: Any, **_kwargs: Any):
        return {"steps": ["stub"], "need_rag": False, "notes": "stub"}

    def _rag_search(*_args: Any, **kwargs: Any):
        seen_queries.append(kwargs.get("query", ""))
        return []

    monkeypatch.setattr(graph, "load_table", _load_table)
    monkeypatch.setattr(graph, "load_passages", _load_passages)
    monkeypatch.setattr(graph, "plan_question", _plan_question)
    monkeypatch.setattr(graph, "rag_search", _rag_search)

    client = TestClient(app)
    payload = {
        "question": "What is the birthplace of the first player in the table, and which country is that city in?",
        "table_id": "test_table",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    assert seen_queries
    assert not re.match(r"^\d", seen_queries[0].strip())


def test_rag_query_uses_city_link(monkeypatch):
    df = pd.DataFrame([{"Name": "APEX Museum", "Area": "Sweet Auburn"}])
    df.attrs["cell_links"] = {(0, "Area"): ["/wiki/Sweet_Auburn"]}
    df.attrs["header_links"] = {}
    seen_queries: list[str] = []

    def _load_table(*_args: Any, **_kwargs: Any):
        return "test_table", df

    def _load_passages(*_args: Any, **_kwargs: Any):
        return []

    def _plan_question(*_args: Any, **_kwargs: Any):
        return {"steps": ["stub"], "need_rag": False, "notes": "stub"}

    def _rag_search(*_args: Any, **kwargs: Any):
        seen_queries.append(kwargs.get("query", ""))
        return []

    monkeypatch.setattr(graph, "load_table", _load_table)
    monkeypatch.setattr(graph, "load_passages", _load_passages)
    monkeypatch.setattr(graph, "plan_question", _plan_question)
    monkeypatch.setattr(graph, "rag_search", _rag_search)

    client = TestClient(app)
    payload = {
        "question": "What country is the city linked in the first row located in?",
        "table_id": "test_table",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    assert seen_queries
    assert "Sweet Auburn" in seen_queries[0]
    assert "APEX Museum" not in seen_queries[0]


def test_birthplace_extraction_born_colon_reasoning(monkeypatch):
    df = pd.DataFrame([{"Player": "Emmitt Smith"}])
    df.attrs["cell_links"] = {(0, "Player"): ["/wiki/Emmitt_Smith"]}
    df.attrs["header_links"] = {}

    def _load_table(*_args: Any, **_kwargs: Any):
        return "test_table", df

    def _load_passages(*_args: Any, **_kwargs: Any):
        return []

    def _plan_question(*_args: Any, **_kwargs: Any):
        return {"steps": ["stub"], "need_rag": False, "notes": "stub"}

    def _rag_search(*_args: Any, **kwargs: Any):
        return [
            Document(
                page_content="Born: May 15, 1969, Pensacola, Florida, U.S.",
                metadata={"source": "passage", "url": "/wiki/Emmitt_Smith"},
            )
        ]

    monkeypatch.setattr(graph, "load_table", _load_table)
    monkeypatch.setattr(graph, "load_passages", _load_passages)
    monkeypatch.setattr(graph, "plan_question", _plan_question)
    monkeypatch.setattr(graph, "rag_search", _rag_search)

    client = TestClient(app)
    payload = {
        "question": "What is the birthplace of the first player in the table, and which country is that city in?",
        "table_id": "test_table",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    calls = response.json()["payload"]["function_calls"]
    assert any(call["name"] == "extract_intermediate" for call in calls)
    reasoning = response.json()["payload"]["reasoning"].lower()
    assert "table provides" not in reasoning


def test_extract_country_prefers_united_states():
    docs = [
        Document(
            page_content="Sweet Auburn is a neighborhood in Atlanta, Georgia, United States.",
            metadata={"source": "passage", "url": "/wiki/Sweet_Auburn"},
        )
    ]
    assert graph._extract_country_from_docs(docs) == "United States"


def test_birthplace_fallback_rag_search(monkeypatch):
    df = pd.DataFrame([{"Player": "Emmitt Smith"}])
    df.attrs["cell_links"] = {(0, "Player"): ["/wiki/Emmitt_Smith"]}
    df.attrs["header_links"] = {}

    def _load_table(*_args: Any, **_kwargs: Any):
        return "test_table", df

    def _load_passages(*_args: Any, **_kwargs: Any):
        return []

    def _plan_question(*_args: Any, **_kwargs: Any):
        return {"steps": ["stub"], "need_rag": False, "notes": "stub"}

    def _rag_search(*_args: Any, **kwargs: Any):
        query = kwargs.get("query", "")
        if "Born:" in query:
            return [
                Document(
                    page_content="Born: May 15, 1969, Pensacola, Florida, U.S.",
                    metadata={"source": "passage", "url": "/wiki/Emmitt_Smith"},
                )
            ]
        if "country" in query:
            return [
                Document(
                    page_content="Pensacola is a city in the United States.",
                    metadata={"source": "passage", "url": "/wiki/Pensacola"},
                )
            ]
        return [
            Document(
                page_content="Emmitt Smith is a former NFL running back.",
                metadata={"source": "passage", "url": "/wiki/Emmitt_Smith"},
            )
        ]

    monkeypatch.setattr(graph, "load_table", _load_table)
    monkeypatch.setattr(graph, "load_passages", _load_passages)
    monkeypatch.setattr(graph, "plan_question", _plan_question)
    monkeypatch.setattr(graph, "rag_search", _rag_search)

    client = TestClient(app)
    payload = {
        "question": "What is the birthplace of the first player in the table, and which country is that city in?",
        "table_id": "test_table",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    calls = response.json()["payload"]["function_calls"]
    assert any(
        call["name"] == "rag_search" and call["arguments"].get("fallback") == "birthplace"
        for call in calls
    )
    assert any(call["name"] == "extract_intermediate" for call in calls)
    final_answer = response.json()["payload"]["final_answer"]
    assert "Pensacola" in final_answer
    assert "United States" in final_answer


def test_birthplace_extracted_outside_top_k(monkeypatch):
    df = pd.DataFrame([{"Player": "Emmitt Smith"}])
    df.attrs["cell_links"] = {(0, "Player"): ["/wiki/Emmitt_Smith"]}
    df.attrs["header_links"] = {}

    def _load_table(*_args: Any, **_kwargs: Any):
        return "test_table", df

    def _load_passages(*_args: Any, **_kwargs: Any):
        return []

    def _plan_question(*_args: Any, **_kwargs: Any):
        return {"steps": ["stub"], "need_rag": False, "notes": "stub"}

    def _rag_search(*_args: Any, **kwargs: Any):
        query = kwargs.get("query", "")
        if "country" in query:
            return [
                Document(
                    page_content="Pensacola is a city in the United States.",
                    metadata={"source": "passage", "url": "/wiki/Pensacola"},
                )
            ]
        noise = [
            Document(page_content=f"noise {idx}", metadata={"source": "passage"})
            for idx in range(6)
        ]
        noise.append(
            Document(
                page_content="Born: May 15, 1969, Pensacola, Florida, U.S.",
                metadata={"source": "passage", "url": "/wiki/Emmitt_Smith"},
            )
        )
        return noise

    monkeypatch.setattr(graph, "load_table", _load_table)
    monkeypatch.setattr(graph, "load_passages", _load_passages)
    monkeypatch.setattr(graph, "plan_question", _plan_question)
    monkeypatch.setattr(graph, "rag_search", _rag_search)

    client = TestClient(app)
    payload = {
        "question": "What is the birthplace of the first player in the table, and which country is that city in?",
        "table_id": "test_table",
        "rag_k": 2,
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    calls = response.json()["payload"]["function_calls"]
    assert any(call["name"] == "rag_search" and call["arguments"].get("hop") == 1 for call in calls)
    assert any(call["name"] == "extract_intermediate" for call in calls)


def test_birthplace_missing_returns_unknown(monkeypatch):
    df = pd.DataFrame([{"Player": "Emmitt Smith"}])
    df.attrs["cell_links"] = {(0, "Player"): ["/wiki/Emmitt_Smith"]}
    df.attrs["header_links"] = {}

    def _load_table(*_args: Any, **_kwargs: Any):
        return "test_table", df

    def _load_passages(*_args: Any, **_kwargs: Any):
        return []

    def _plan_question(*_args: Any, **_kwargs: Any):
        return {"steps": ["stub"], "need_rag": False, "notes": "stub"}

    def _rag_search(*_args: Any, **_kwargs: Any):
        return [
            Document(
                page_content="Emmitt Smith is a former NFL running back.",
                metadata={"source": "passage", "url": "/wiki/Emmitt_Smith"},
            )
        ]

    monkeypatch.setattr(graph, "load_table", _load_table)
    monkeypatch.setattr(graph, "load_passages", _load_passages)
    monkeypatch.setattr(graph, "plan_question", _plan_question)
    monkeypatch.setattr(graph, "rag_search", _rag_search)

    client = TestClient(app)
    payload = {
        "question": "What is the birthplace of the first player in the table, and which country is that city in?",
        "table_id": "test_table",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()["payload"]
    assert data["final_answer"] == "Unknown"
    assert data["final_answer_structured"]["error"] == "not_found_in_passages"


def test_debug_includes_rag_debug(monkeypatch):
    df = pd.DataFrame([{"Player": "Alice"}])
    df.attrs["cell_links"] = {(0, "Player"): ["/wiki/Alice"]}
    df.attrs["header_links"] = {}

    def _load_table(*_args: Any, **_kwargs: Any):
        return "test_table", df

    def _load_passages(*_args: Any, **_kwargs: Any):
        return []

    def _plan_question(*_args: Any, **_kwargs: Any):
        return {"steps": ["stub"], "need_rag": True, "notes": "stub"}

    def _rag_search(*_args: Any, **_kwargs: Any):
        return [
            Document(
                page_content="Alice was born in Paris, France.",
                metadata={"source": "passage", "url": "/wiki/Alice", "title": "Alice"},
            )
        ]

    monkeypatch.setattr(graph, "load_table", _load_table)
    monkeypatch.setattr(graph, "load_passages", _load_passages)
    monkeypatch.setattr(graph, "plan_question", _plan_question)
    monkeypatch.setattr(graph, "rag_search", _rag_search)

    client = TestClient(app)
    payload = {
        "question": "What is the birthplace of the first player in the table, and which country is that city in?",
        "table_id": "test_table",
        "debug": True,
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    rag_debug = response.json()["payload"].get("rag_debug")
    assert rag_debug
    assert rag_debug[0]["docs"]
    assert rag_debug[0]["docs"][0]["preview"]


def test_passage_lookup_birthplace(monkeypatch):
    df = pd.DataFrame([{"Player": "Alice"}])
    df.attrs["cell_links"] = {(0, "Player"): ["/wiki/Alice"]}
    df.attrs["header_links"] = {}

    def _load_table(*_args: Any, **_kwargs: Any):
        return "test_table", df

    def _load_passages(*_args: Any, **_kwargs: Any):
        return [
            {
                "url": "/wiki/Alice",
                "title": "Alice",
                "text": "Alice was born in Paris, France.",
            }
        ]

    def _plan_question(*_args: Any, **_kwargs: Any):
        return {"steps": ["stub"], "need_rag": False, "notes": "stub"}

    def _rag_search(*_args: Any, **kwargs: Any):
        query = kwargs.get("query", "")
        if "country" in query:
            return [
                Document(
                    page_content="Paris is a city in France.",
                    metadata={"source": "passage", "url": "/wiki/Paris"},
                )
            ]
        return []

    monkeypatch.setattr(graph, "load_table", _load_table)
    monkeypatch.setattr(graph, "load_passages", _load_passages)
    monkeypatch.setattr(graph, "plan_question", _plan_question)
    monkeypatch.setattr(graph, "rag_search", _rag_search)

    client = TestClient(app)
    payload = {
        "question": "What is the birthplace of the first player in the table, and which country is that city in?",
        "table_id": "test_table",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    calls = response.json()["payload"]["function_calls"]
    assert any(call["name"] == "passage_lookup" for call in calls)
    assert "France" in response.json()["payload"]["final_answer"]


def test_berlin_marathon_city_cleaning(monkeypatch):
    df = pd.DataFrame([{"Runner": "Alice"}])
    df.attrs["cell_links"] = {(0, "Runner"): ["/wiki/Alice"]}
    df.attrs["header_links"] = {}
    seen_queries: list[str] = []

    def _load_table(*_args: Any, **_kwargs: Any):
        return "test_table", df

    def _load_passages(*_args: Any, **_kwargs: Any):
        return []

    def _plan_question(*_args: Any, **_kwargs: Any):
        return {"steps": ["stub"], "need_rag": False, "notes": "stub"}

    def _rag_search(*_args: Any, **kwargs: Any):
        query = kwargs.get("query", "")
        seen_queries.append(query)
        if "country" in query:
            return [
                Document(
                    page_content="Nakuru is a city in Kenya.",
                    metadata={"source": "passage", "url": "/wiki/Nakuru"},
                )
            ]
        return [
            Document(
                page_content="She was born in Nakuru, Kenya.",
                metadata={"source": "passage", "url": "/wiki/Alice"},
            )
        ]

    monkeypatch.setattr(graph, "load_table", _load_table)
    monkeypatch.setattr(graph, "load_passages", _load_passages)
    monkeypatch.setattr(graph, "plan_question", _plan_question)
    monkeypatch.setattr(graph, "rag_search", _rag_search)

    client = TestClient(app)
    payload = {
        "question": "What is the birthplace of the first runner in the table, and which country is that city in?",
        "table_id": "test_table",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    assert any("Nakuru" in query and "in Nakuru" not in query for query in seen_queries if "country" in query)


def test_rubens_alias_rejected(monkeypatch):
    df = pd.DataFrame([{"Driver": "Rubens Barrichello"}])
    df.attrs["cell_links"] = {(0, "Driver"): ["/wiki/Rubens_Barrichello"]}
    df.attrs["header_links"] = {}
    seen_queries: list[str] = []

    def _load_table(*_args: Any, **_kwargs: Any):
        return "test_table", df

    def _load_passages(*_args: Any, **_kwargs: Any):
        return []

    def _plan_question(*_args: Any, **_kwargs: Any):
        return {"steps": ["stub"], "need_rag": False, "notes": "stub"}

    def _rag_search(*_args: Any, **kwargs: Any):
        query = kwargs.get("query", "")
        seen_queries.append(query)
        if "country" in query:
            return [
                Document(
                    page_content="Sao Paulo is a city in Brazil.",
                    metadata={"source": "passage", "url": "/wiki/Sao_Paulo"},
                )
            ]
        return [
            Document(
                page_content="Rubens Barrichello (also known as Rubinho) was born in Sao Paulo, Brazil.",
                metadata={"source": "passage", "url": "/wiki/Rubens_Barrichello"},
            )
        ]

    monkeypatch.setattr(graph, "load_table", _load_table)
    monkeypatch.setattr(graph, "load_passages", _load_passages)
    monkeypatch.setattr(graph, "plan_question", _plan_question)
    monkeypatch.setattr(graph, "rag_search", _rag_search)

    client = TestClient(app)
    payload = {
        "question": "What is the birthplace of the first driver in the table, and which country is that city in?",
        "table_id": "test_table",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    assert all("also known as" not in query for query in seen_queries)
    assert any("Sao Paulo" in query for query in seen_queries if "country" in query)


def test_emmitt_birthplace_hop1(monkeypatch):
    df = pd.DataFrame([{"Player": "Emmitt Smith"}])
    df.attrs["cell_links"] = {(0, "Player"): ["/wiki/Emmitt_Smith"]}
    df.attrs["header_links"] = {}

    def _load_table(*_args: Any, **_kwargs: Any):
        return "test_table", df

    def _load_passages(*_args: Any, **_kwargs: Any):
        return []

    def _plan_question(*_args: Any, **_kwargs: Any):
        return {"steps": ["stub"], "need_rag": False, "notes": "stub"}

    def _rag_search(*_args: Any, **kwargs: Any):
        query = kwargs.get("query", "")
        if "country" in query:
            return [
                Document(
                    page_content="Pensacola is a city in the United States.",
                    metadata={"source": "passage", "url": "/wiki/Pensacola"},
                )
            ]
        return [
            Document(
                page_content="place of birth: Pensacola, Florida.",
                metadata={"source": "passage", "url": "/wiki/Emmitt_Smith"},
            )
        ]

    monkeypatch.setattr(graph, "load_table", _load_table)
    monkeypatch.setattr(graph, "load_passages", _load_passages)
    monkeypatch.setattr(graph, "plan_question", _plan_question)
    monkeypatch.setattr(graph, "rag_search", _rag_search)

    client = TestClient(app)
    payload = {
        "question": "What is the birthplace of the first player in the table, and which country is that city in?",
        "table_id": "test_table",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    calls = response.json()["payload"]["function_calls"]
    hops = {call["arguments"].get("hop") for call in calls if call["name"] == "rag_search"}
    assert 0 in hops and 1 in hops
    assert "United States" in response.json()["payload"]["final_answer"]


def test_museum_country_only(monkeypatch):
    df = pd.DataFrame([{"Name": "APEX Museum", "Area": "Sweet Auburn"}])
    df.attrs["cell_links"] = {(0, "Area"): ["/wiki/Sweet_Auburn"]}
    df.attrs["header_links"] = {}

    def _load_table(*_args: Any, **_kwargs: Any):
        return "test_table", df

    def _load_passages(*_args: Any, **_kwargs: Any):
        return []

    def _plan_question(*_args: Any, **_kwargs: Any):
        return {"steps": ["stub"], "need_rag": False, "notes": "stub"}

    def _rag_search(*_args: Any, **_kwargs: Any):
        return [
            Document(
                page_content="Sweet Auburn is a neighborhood in Atlanta, Georgia, United States.",
                metadata={"source": "passage", "url": "/wiki/Sweet_Auburn"},
            )
        ]

    monkeypatch.setattr(graph, "load_table", _load_table)
    monkeypatch.setattr(graph, "load_passages", _load_passages)
    monkeypatch.setattr(graph, "plan_question", _plan_question)
    monkeypatch.setattr(graph, "rag_search", _rag_search)

    client = TestClient(app)
    payload = {
        "question": "What country is the city linked in the first row located in? Answer with country only.",
        "table_id": "test_table",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    assert response.json()["payload"]["final_answer"] == "United States"
