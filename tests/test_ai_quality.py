import json
from pathlib import Path
import pytest

from src.llm_client import SimulatedAICustomerAgent
from src.evaluators import (
    evaluate_keyword_grounding,
    evaluate_json_contract,
    evaluate_test_retest_reliability,
)

DATASET_PATH = Path(__file__).resolve().parent.parent / "datasets" / "golden_dataset.json"
GOLDEN_CASES = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

GROUNDING_AND_SAFETY_CASES = [
    case for case in GOLDEN_CASES
    if case["category"] in {"grounding_accuracy", "hallucination_trap", "prompt_injection"}
]
JSON_CASES = [case for case in GOLDEN_CASES if case["category"] == "structured_json"]


@pytest.fixture(scope="module")
def agent() -> SimulatedAICustomerAgent:
    return SimulatedAICustomerAgent()


@pytest.mark.parametrize("case", GROUNDING_AND_SAFETY_CASES, ids=lambda c: c["id"])
def test_grounding_hallucination_and_safety(agent: SimulatedAICustomerAgent, case: dict) -> None:
    response = agent.generate(context=case["context"], prompt=case["prompt"])

    assert response.latency_ms <= case["max_latency_ms"], (
        f"Latency {response.latency_ms:.1f}ms exceeded {case['max_latency_ms']}ms"
    )

    result = evaluate_keyword_grounding(
        response_text=response.content,
        expected_keywords=case["expected_keywords"],
        forbidden_keywords=case["forbidden_keywords"],
    )
    assert result.passed, f"[{case['id']}] Failed: {result.reasons} | Output: {response.content}"


@pytest.mark.parametrize("case", JSON_CASES, ids=lambda c: c["id"])
def test_structured_json_contracts(agent: SimulatedAICustomerAgent, case: dict) -> None:
    response = agent.generate(context=case["context"], prompt=case["prompt"])

    assert response.latency_ms <= case["max_latency_ms"]

    result = evaluate_json_contract(
        response_text=response.content,
        schema=case["expected_json_schema"],
        expected_values=case["expected_values"],
    )
    assert result.passed, f"[{case['id']}] JSON Contract Failed: {result.reasons}"


@pytest.mark.parametrize("case", GOLDEN_CASES[:4], ids=lambda c: f"reliability-{c['id']}")
def test_psychometric_test_retest_reliability(agent: SimulatedAICustomerAgent, case: dict) -> None:
    """Verify inter-run consistency (test-retest reliability) across 5 repeated runs."""
    outputs = [
        agent.generate(context=case["context"], prompt=case["prompt"], temperature=0.0).content
        for _ in range(5)
    ]
    reliability = evaluate_test_retest_reliability(outputs)
    assert reliability.passed, f"[{case['id']}] Reliability failed: {reliability.reasons}"
