# AI / LLM Quality Evaluation Suite (Psychometric QA)

[![AI Quality Evaluation Suite](https://github.com/MrMoon1725/ai-llm-qa-evaluation/actions/workflows/ai-eval.yml/badge.svg)](https://github.com/MrMoon1725/ai-llm-qa-evaluation/actions/workflows/ai-eval.yml)

Automated quality assurance and evaluation framework for **AI / LLM-powered applications**, built with **Python**, **Pytest**, and **JSON Schema**. It bridges traditional software QA with **psychometric evaluation principles** (construct validity, test-retest reliability, hallucination prevention, and prompt-injection resilience).

---

## Why Psychometric QA for AI Products?

Unlike deterministic software, LLM outputs are probabilistic. Drawing on **psychometrics** and **software testing**, this suite measures:

1. **Construct Validity (`evaluate_keyword_grounding`):** Does the model answer strictly from grounded context without fabricating facts?
2. **Test-Retest Reliability (`evaluate_test_retest_reliability`):** Does the model produce consistent outputs across repeated identical executions?
3. **Contract Compliance (`evaluate_json_contract`):** Do structured outputs strictly adhere to JSON Schemas (`jsonschema`)?
4. **Adversarial Safety & Resilience:** Does the model resist prompt injection, system prompt leakage, and unauthorized policy overrides?

---

## Project Structure

```text
ai-llm-qa-evaluation/
├── .github/workflows/
│   └── ai-eval.yml             # CI pipeline executing Pytest & uploading HTML report
├── datasets/
│   └── golden_dataset.json     # Curated evaluation cases (grounding, JSON, traps, injection)
├── src/
│   ├── llm_client.py           # Swappable LLM client interface + deterministic CI simulator
│   └── evaluators.py           # Grounding, JSON schema, & test-retest reliability evaluators
├── tests/
│   └── test_ai_quality.py      # 12 parametrized automated evaluation tests
├── requirements.txt
└── README.md
```

---

## Running Locally in a Controlled Environment

```bash
git clone https://github.com/MrMoon1725/ai-llm-qa-evaluation.git
cd ai-llm-qa-evaluation
python -m venv .venv
# Windows: .venv\Scripts\activate  |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest -v --html=reports/ai-eval-report.html --self-contained-html
```
