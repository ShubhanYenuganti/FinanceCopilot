import streamlit as st
import plotly.express as px
import os
import pandas as pd

from agent.metrics import load_fixtures
from agent.router import execute_intent
from agent.llm_agent import CFOAgent

st.set_page_config(page_title="CFO Copilot", layout="wide")
st.title("CFO Copilot - OpenAI")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
agent = CFOAgent(api_key=OPENAI_API_KEY, model="gpt-4o-mini")

DATA = load_fixtures()

q = st.chat_input("Ask a question.")
if q:
    with st.spinner("Classifying..."):
        intent = agent.classify(q) 
    st.write("**Intent:**", intent)

    with st.spinner("Computing KPI..."):
        df = execute_intent(DATA, intent)
    
    print(df)
        
    with st.spinner("Summarizing..."):
        summary = agent.narrate(q, intent, df.to_dict(orient="records"))

    st.subheader("Answer")
    st.markdown(summary)

    kind = intent["kind"]
    if kind == "revenue_vs_budget" and not df.empty:
        df["month"] = pd.to_datetime(df["month"]).dt.strftime("%b %Y")
        df = df.rename(columns = {
            "revenue_actual_usd": "Revenue (Actual)",
            "revenue_budget_usd": "Revenue (Budget)"
        })
        fig = px.bar(df, x="month", y=["Revenue (Actual)","Revenue (Budget)"], barmode="group", title="Revenue vs Budget")
        st.plotly_chart(fig, use_container_width=True)
    elif kind == "gross_margin_pct" and not df.empty:
        df = df.rename(columns = {
            "gross_margin_pct": "Gross Margin %"
        })
        fig = px.line(df, x="month", y="Gross Margin %", markers=True)
        fig.update_xaxes(tickformat="%b %Y")
        st.plotly_chart(fig, use_container_width=True)
    elif kind == "opex_breakdown" and not df.empty:
        df = df.rename(columns = {
            "amount_usd": "$ (usd)"
        })
        fig = px.pie(df, names="category", values="$ (usd)")
        st.plotly_chart(fig, use_container_width=True)
    elif kind == "ebitda_proxy" and not df.empty:
        df_long = df.melt(
            id_vars="month",
            value_vars=["revenue_usd", "cogs_usd", "opex_total_usd"],
            var_name="category",
            value_name="amount"
        )
        fig = px.bar(
            df_long,
            x="month",
            y="amount",
            color="category",
            barmode="group",
            title="Revenue vs COGS vs Opex by Month"
        )
        fig.update_xaxes(tickformat="%b %Y")

        st.plotly_chart(fig, use_container_width=True)
    elif kind == "cash_runway" and not df.empty:
        df_melted = df.melt(
            value_vars = ["latest_cash_usd", "avg_burn_last3m_usd", "avg_buffer_last3m_usd"],
            var_name = "Metric",
            value_name = "USD"
        )
        
        df_melted["Metric"] = df_melted["Metric"].replace({
            "latest_cash_usd": "Latest Cash (USD)",
            "avg_burn_last3m_usd": "Avg Burn (Last 3 Months) (USD)",
            "avg_buffer_last3m_usd": "Avg Buffer (Last 3 Months) (USD)"
        })
        
        fig = px.bar(
            df_melted,
            x="Metric",
            y="USD",
            text = "USD",
            title="Cash Runway Metrics",
            labels={"USD": "Amount (USD)"}
        )
        
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig.update_layout(yaxis_tickformat=",", uniformtext_minsize=8, uniformtext_mode='hide')

        st.plotly_chart(fig, use_container_width=True)
        
        