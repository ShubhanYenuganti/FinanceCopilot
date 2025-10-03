import pandas as pd
import numpy as np

def load_fixtures(
    actuals_csv = "fixtures/actuals_m.csv",
    budget_csv = "fixtures/budget_m.csv",
    cash_csv = "fixtures/cash_m.csv",
    fx_csv = "fixtures/fx_m.csv"
):
    actuals = pd.read_csv(actuals_csv)
    budget = pd.read_csv(budget_csv)
    cash = pd.read_csv(cash_csv)
    fx = pd.read_csv(fx_csv)
    
    for df in (actuals, budget, cash, fx):
        df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M").astype(str)
    
    # numerics
    actuals["amount"] = pd.to_numeric(actuals["amount"], errors="coerce").fillna(0.0)
    budget["amount"] = pd.to_numeric(budget["amount"], errors="coerce").fillna(0.0)
    actuals["amount_usd"] = pd.to_numeric(actuals["amount_usd"], errors="coerce").fillna(0.0)
    budget["amount_usd"] = pd.to_numeric(budget["amount_usd"], errors="coerce").fillna(0.0)
    cash["cash_usd"] = pd.to_numeric(cash["cash_usd"], errors="coerce")
    fx["rate_to_usd"] = pd.to_numeric(fx["rate_to_usd"], errors="coerce")
    
    return {"actuals": actuals, "budget": budget, "fx": fx, "cash": cash}

def _filter(df, period_start = None, period_end = None, entity = None):
    out = df
    # keep only rows on or after given starting month
    if period_start:
        out = out[out["month"].astype("period[M]") >= pd.Period(period_start)]
    # keep rows before or on the ending month
    if period_end:
        out = out[out["month"].astype("period[M]") <= pd.Period(period_end)]
    if entity and "entity" in out.columns:
        out = out[out["entity"].astype(str).str.lower() == str(entity).lower()]
    return out

def revenue_vs_budget(data, period_start=None, period_end = None, entity = None):
    """
    Returns monthly Revenue Actual vs Budget with variance in USD.
    Columns: month(str), revenue_actual_usd, revenue_budget_usd, variance_usd
    """
    
    
    a = _filter(data["actuals"], period_start, period_end, entity)
    b = _filter(data["budget"], period_start, period_end, entity)
    
    a = a[a["account_category"].str.lower() == "revenue"]
    b = b[b["account_category"].str.lower() == "revenue"]
    
    a_m = a.groupby("month", as_index = False)["amount_usd"].sum().rename(columns={"amount_usd":"revenue_actual_usd"})
    b_m = b.groupby("month", as_index = False)["amount_usd"].sum().rename(columns={"amount_usd":"revenue_budget_usd"})

    out = a_m.merge(b_m, on="month", how="outer").fillna(0.0).sort_values("month")
    out["variance_usd"] = out["revenue_actual_usd"] - out["revenue_budget_usd"]
    out["month"] = out["month"].astype(str)
    
    return out

def gross_margin_pct(data, period_start=None, period_end = None, entity = None):
    """
    GM% = (Revenue - COGS) / Revenue
    """
    
    df = _filter(data["actuals"], period_start, period_end, entity)
    
    rev = df[df["account_category"].eq("Revenue")].groupby("month", as_index = False)["amount_usd"].sum().rename(columns={"amount_usd":"revenue_usd"})
    cogs = df[df["account_category"].eq("COGS")].groupby("month", as_index = False)["amount_usd"].sum().rename(columns={"amount_usd":"cogs_usd"})
    
    out = rev.merge(cogs, on="month", how = "outer").fillna(0.0).sort_values("month")
    out["gross_margin_pct"] = np.where(out["revenue_usd"] > 0, (out["revenue_usd"] - out["cogs_usd"]) / out["revenue_usd"], 0.0)
    out["month"] = out["month"].astype(str)
    
    print(out)

    return out

def opex_breakdown(data, month, entity=None):
    """
    Single month Opex breakdown grouped by category after 'Opex:'.
    """
    
    df = data["actuals"]
    
    df = df[df["month"].astype("period[M]") == pd.Period(month)] 
    
    if entity:
        df = df[df["entity"].astype(str).str.lower() == str(entity).lower()]
        
    opex = df[df["account_category"].str.lower().str.startswith("opex:")].copy()
    if opex.empty:
        return pd.DataFrame(columns=["category", "amount_usd"])
    
    opex["category"] = opex["account_category"].str.split(":", n = 1, expand = True)[1]
    out = opex.groupby("category", as_index = False)["amount_usd"].sum().sort_values("amount_usd", ascending=False)
    
    return out

def ebitda_proxy(data, period_start = None, period_end = None, entity = None):
    """
    EBITDA (proxy) = Revenue - COGS - Opex total
    """
    
    df = _filter(data["actuals"], period_start, period_end, entity)
    df["acct_l"] = df["account_category"].str.lower()

    pivot = df.pivot_table(index="month", columns="acct_l", values="amount_usd", aggfunc="sum", fill_value=0.0)
    rev  = pivot.get("revenue", 0.0)
    cogs = pivot.get("cogs", 0.0)
    opex_cols = [c for c in pivot.columns if c.startswith("opex:")]
    opex_total = pivot[opex_cols].sum(axis=1) if opex_cols else 0.0

    ebitda = rev - cogs - opex_total

    out = pd.DataFrame({
        "month": pivot.index.astype(str),
        "revenue_usd": rev.values,
        "cogs_usd": cogs.values,
        "opex_total_usd": opex_total.values,
        "ebitda_proxy_usd": ebitda.values
    }).sort_values("month")

    return out

def cash_runway(data, period_start = None, period_end = None, entity = None):
    """
    Runway - latest cash / avg monthly net burn over last 3 months (default)
    net burn = (Opex + COGS - Revenue)
    """
    
    df = _filter(data["actuals"], period_start, period_end, entity)
    df["acct_l"] = df["account_category"].str.lower()

    by_m = df.pivot_table(index="month", columns="acct_l", values="amount_usd", aggfunc="sum", fill_value=0.0).sort_index()
    rev  = by_m.get("revenue", 0.0)
    cogs = by_m.get("cogs", 0.0)
    opex_cols = [c for c in by_m.columns if c.startswith("opex:")]
    opex = by_m[opex_cols].sum(axis=1) if opex_cols else 0.0
    net_burn = (opex + cogs - rev)
    
    avg_last3 = float(net_burn.tail(3).mean()) if len(net_burn) else 0.0
    
    # Interpret sign
    if avg_last3 >= 0:
        burn = avg_last3
        buffer = 0.0
        burn_or_buffer = "burn"
    else:
        burn = 0.0
        buffer = abs(avg_last3)  
        burn_or_buffer = "buffer"
    
    if period_start is not None:
        cash_df = _filter(data["cash"], period_start, period_end, entity)
        latest_cash = cash_df["cash_usd"].iloc[-1] if len(cash_df) else 0.0
    else:
        cash_df = data["cash"].sort_values("month")
        latest_cash = cash_df["cash_usd"].iloc[-1] if len(cash_df) else 0.0

    runway_months = (latest_cash / burn) if burn > 0 else float("inf")
    out = pd.DataFrame([{
        "runway_months": runway_months,
        "latest_cash_usd": latest_cash,
        "avg_burn_last3m_usd": burn,         
        "avg_buffer_last3m_usd": buffer,     
        "burn_or_buffer": burn_or_buffer     
    }])
    
    return out