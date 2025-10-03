import os
import pandas as pd
import pytest
import math


from agent.metrics import load_fixtures
from agent.router import execute_intent
from agent.llm_agent import CFOAgent

@pytest.fixture(scope="session")
def agent():
    api_key = os.environ.get("OPENAI_API_KEY")
    return CFOAgent(api_key=api_key, model="gpt-4o-mini")

@pytest.fixture(scope="session")
def data():
    return load_fixtures()

# Test 1 Revenue vs Budget
def test_revenue_vs_budget(agent, data):
    q = "What was the revenue vs budget in June 2025?"
    intent = agent.classify(q)
    assert intent["kind"] == "revenue_vs_budget"
    assert intent["month"] == "2025-06"
    
    df = execute_intent(data, intent)
    expected_actual = float(pd.to_numeric(df["revenue_actual_usd"]))
    expected_budget = float(pd.to_numeric(df["revenue_budget_usd"]))
    expected_variance = expected_actual - expected_budget

    assert math.isclose(expected_actual, 1014896.0, rel_tol=1e-9)
    assert math.isclose(expected_budget, 1072687.68, rel_tol=1e-9)
    assert math.isclose(expected_variance, 1014896.0 - 1072687.68, rel_tol=1e-9)

    summary = agent.narrate(q, intent, df.to_dict(orient="records"))
    
    assert "1014896.0" in summary or "1014896" in summary or "1,014,896.0" in summary or "1,014,896" in summary
    assert "1072687.68" in summary or "1072687.7" in summary or "1072688" in summary or "1,072,687.68" in summary or "1,072,687.7" in summary or "1,072,688" in summary

# Test 2 Opex Breakdown
def test_opex_breakdown_end_to_end(agent, data):
    q = "Show me the Opex breakdown for ParentCo in May 2024."
    intent = agent.classify(q)
    assert intent["kind"] == "opex_breakdown"
    assert intent["month"] == "2024-05"
    assert intent["entity"].lower() == "parentco"
    
    df = execute_intent(data, intent)
    assert df[df["category"] == "Marketing"]["amount_usd"].values[0] == 123000
    assert df[df["category"] == "Sales"]["amount_usd"].values[0] == 73800
    assert df[df["category"] == "R&D"]["amount_usd"].values[0] == 49200
    assert df[df["category"] == "Admin"]["amount_usd"].values[0] == 36900

    summary = agent.narrate(q, intent, df.to_dict(orient="records"))
    assert "123000" in summary or "123,000" in summary
    assert "73800" in summary or "73,800" in summary
    assert "49200" in summary or "49,200" in summary
    assert "36900" in summary or "36,900" in summary
    assert "Marketing" in summary
    assert "Sales" in summary
    assert "R&D" in summary
    assert "Admin" in summary
    