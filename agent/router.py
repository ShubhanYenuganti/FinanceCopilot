import pandas as pd
from agent.metrics import (
    revenue_vs_budget,
    gross_margin_pct,
    opex_breakdown,
    ebitda_proxy,
    cash_runway,
)

def execute_intent(data, intent: dict) -> pd.DataFrame:
    kind = intent.get("kind")
    if intent.get("month") is not None:
        ps = intent.get("month")
        pe = intent.get("month")
    else:
        ps   = intent.get("period_start")
        pe   = intent.get("period_end")
    ent  = intent.get("entity")

    if kind == "revenue_vs_budget":
        return revenue_vs_budget(data, ps, pe, ent)
    if kind == "gross_margin_pct":
        return gross_margin_pct(data, ps, pe, ent)
    if kind == "opex_breakdown":
        month = intent.get("month") or pe or ps
        return opex_breakdown(data, month, ent)
    if kind == "ebitda_proxy":
        return ebitda_proxy(data, ps, pe, ent)
    if kind == "cash_runway":
        return cash_runway(data, ps, pe, ent)
    raise ValueError(f"Unknown intent: {kind}")