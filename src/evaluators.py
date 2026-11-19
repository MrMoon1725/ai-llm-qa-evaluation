import json
from dataclasses import dataclass
from typing import Any, Dict, List
import jsonschema


@dataclass
class EvaluationResult:
    passed: bool
    score: float
    reasons: List[str]


def evaluate_keyword_grounding(
    response_text: str,
    expected_keywords: List[str],
    forbidden_keywords: List[str],
) -> EvaluationResult:
    """
    Psychometric Construct Validity check: verifies the response contains
    grounded facts and avoids hallucinated or unsafe forbidden terms.
    """
    lower_resp = response_text.lower()
    reasons: List[str] = []

    matched_expected = [kw for kw in expected_keywords if kw.lower() in lower_resp]
    if not matched_expected:
        reasons.append(f"Missing expected keywords: {expected_keywords}")

    violated_forbidden = [kw for kw in forbidden_keywords if kw.lower() in lower_resp]
    if violated_forbidden:
        reasons.append(f"Detected forbidden/hallucinated terms: {violated_forbidden}")

    score = len(matched_expected) / max(len(expected_keywords), 1)
    if violated_forbidden:
        score = 0.0

    passed = len(matched_expected) > 0 and len(violated_forbidden) == 0
    return EvaluationResult(passed=passed, score=round(score, 2), reasons=reasons)


def evaluate_json_contract(
    response_text: str,
    schema: Dict[str, Any],
    expected_values: Dict[str, Any],
) -> EvaluationResult:
    """Validates strict JSON schema compliance and expected classification values."""
    reasons: List[str] = []
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        return EvaluationResult(passed=False, score=0.0, reasons=[f"Invalid JSON: {exc}"])

    try:
        jsonschema.validate(instance=parsed, schema=schema)
    except jsonschema.ValidationError as exc:
        return EvaluationResult(passed=False, score=0.0, reasons=[f"Schema violation: {exc.message}"])

    mismatches = [
        f"{k}: expected {v}, got {parsed.get(k)}"
        for k, v in expected_values.items()
        if parsed.get(k) != v
    ]
    if mismatches:
        reasons.extend(mismatches)
        return EvaluationResult(passed=False, score=0.5, reasons=reasons)

    return EvaluationResult(passed=True, score=1.0, reasons=[])


def evaluate_test_retest_reliability(outputs: List[str]) -> EvaluationResult:
    """
    Psychometric Test-Retest Reliability check: measures response stability
    across N repeated executions of the exact same prompt.
    """
    if not outputs:
        return EvaluationResult(passed=False, score=0.0, reasons=["No outputs provided"])

    unique_outputs = set(outputs)
    consistency_ratio = 1.0 - ((len(unique_outputs) - 1) / max(len(outputs), 1))
    passed = len(unique_outputs) == 1
    reasons = [] if passed else [f"Inconsistent outputs across {len(outputs)} runs: {len(unique_outputs)} variants"]
    return EvaluationResult(passed=passed, score=round(consistency_ratio, 2), reasons=reasons)
