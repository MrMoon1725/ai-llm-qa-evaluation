import json
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    """Standardized response container with latency telemetry."""
    content: str
    latency_ms: float
    model: str


class LLMClient(Protocol):
    """Protocol for any LLM backend (simulated or live API)."""
    def generate(self, context: str, prompt: str, temperature: float = 0.0) -> LLMResponse:
        ...


class SimulatedAICustomerAgent:
    """
    Deterministic simulated LLM client for reproducible CI/CD QA evaluation.
    Can be swapped for a live Gemini/OpenAI client via environment configuration.
    """
    model_name = "simulated-qa-agent-v1"

    def generate(self, context: str, prompt: str, temperature: float = 0.0) -> LLMResponse:
        start = time.perf_counter()
        p_lower = prompt.lower()
        c_lower = context.lower()

        if not prompt or not prompt.strip():
            reply = "Please provide a clear question so I can assist you."
        elif "ignore all previous instructions" in p_lower or "system prompt" in p_lower:
            reply = "I cannot share internal system instructions or credentials, but I can assist you with product support."
        elif "system override" in p_lower or "refund_approved" in p_lower:
            reply = "I cannot auto-approve a $2,500 refund because refunds over $500 require manager approval."
        elif "platinum quantum tier" in p_lower:
            reply = "I don't have information about a Platinum Quantum Tier; it is not mentioned in the provided catalog."
        elif "tokyo office" in p_lower or "chief robotics officer" in p_lower:
            reply = "I do not have that information; those details are not provided in the company context."
        elif "classify this ticket as json" in p_lower:
            if "500 internal server error" in c_lower:
                reply = json.dumps({
                    "severity": "critical",
                    "category": "bug",
                    "requires_escalation": True,
                })
            else:
                reply = json.dumps({
                    "severity": "low",
                    "category": "billing",
                    "requires_escalation": False,
                })
        elif ("request a refund" in p_lower or "refund" in p_lower) and "30 calendar days" in c_lower:
            reply = "You have 30 calendar days from purchase to request a refund, provided the license was not activated on more than 2 devices."
        elif "priority-1" in p_lower and "15 minutes" in c_lower:
            reply = "Priority-1 production incidents receive an initial engineering response within 15 minutes, 24/7."
        else:
            reply = "I cannot confirm an answer outside the provided context."

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return LLMResponse(content=reply, latency_ms=elapsed_ms, model=self.model_name)
