import pandas as pd
import numpy as np

actuals_df = pd.read_csv("../fixtures/actuals.csv")
budget_df = pd.read_csv("../fixtures/budget.csv")
cash_df = pd.read_csv("../fixtures/cash.csv")
fx_df = pd.read_csv("../fixtures/fx.csv")

# Month as monthly string "YYYY-MM" (pandas Period -> string)
for df in (actuals_df, budget_df, cash_df, fx_df):
    df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M").astype(str)
    
# Numeric coercions
actuals_df["amount"] = pd.to_numeric(actuals_df["amount"], errors="coerce").fillna(0.0)
budget_df["amount"]  = pd.to_numeric(budget_df["amount"],  errors="coerce").fillna(0.0)
cash_df["cash_usd"]   = pd.to_numeric(cash_df["cash_usd"],   errors="coerce").fillna(0.0)
fx_df["rate_to_usd"] = pd.to_numeric(fx_df["rate_to_usd"], errors="coerce").fillna(1.0)

# FX Conversion
def to_usd(df, value_col):
    fx_small = fx_df[["month","currency","rate_to_usd"]]
    out = df.merge(fx_small, on=["month","currency"], how="left")
    out["rate_to_usd"] = out["rate_to_usd"].fillna(1.0)
    out[value_col + "_usd"] = out[value_col] * out["rate_to_usd"]
    return out

actuals_usd = to_usd(actuals_df, "amount")
budget_usd  = to_usd(budget_df, "amount")

actuals_out = actuals_usd[["month","entity","account_category","amount","currency","amount_usd"]].copy()
budget_out  = budget_usd[ ["month","entity","account_category","amount","currency","amount_usd"]].copy()
fx_out      = fx_df[     ["month","currency","rate_to_usd"]].copy()
cash_out    = cash_df[  ["month","entity","cash_usd"]].copy()

actuals_out.to_csv("../fixtures/actuals_m.csv", index = False)
budget_out.to_csv("../fixtures/budget_m.csv", index=False)
fx_out.to_csv("../fixtures/fx_m.csv", index=False)
cash_out.to_csv("../fixtures/cash_m.csv", index=False)
