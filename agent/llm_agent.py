# agent/llm_agent.py
import json
from openai import OpenAI
import re

INTENT_SCHEMA = {
    "name": "FinanceIntent",
    "schema": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": [
                    "revenue_vs_budget",
                    "gross_margin_pct",
                    "opex_breakdown",
                    "ebitda_proxy",
                    "cash_runway"
                ],
                "description": "Which KPI to compute."
            },
            "period_start": {
                "type": ["string", "null"],
                "pattern": "^[0-9]{4}-(0[1-9]|1[0-2])$",
                "description": "YYYY-MM start month (inclusive)."
            },
            "period_end": {
                "type": ["string", "null"],
                "pattern": "^[0-9]{4}-(0[1-9]|1[0-2])$",
                "description": "YYYY-MM end month (inclusive)."
            },
            "month": {
                "type": ["string", "null"],
                "pattern": "^[0-9]{4}-(0[1-9]|1[0-2])$",
                "description": "YYYY-MM for single-month views (e.g., Opex breakdown)."
            },
            "entity": {
                "type": ["string", "null"],
                "description": "Optional entity/business unit filter. Case-insensitive match on 'entity' column."
            }
        },
        "required": ["kind"],
        "additionalProperties": False
    }
}

SYSTEM_GUIDE = (
    "You classify CFO questions about monthly financials into a strict JSON object called FinanceIntent. "
    "Choose exactly one 'kind' from the enum. "
    "Infer period_start/period_end/month when possible from natural language (e.g. 'June 2025' -> '2025-06'). "
    "If the user asks for 'last N months', set period_end to the latest month implied in the question or leave null; "
    "set period_start to N months before if you can. "
    "If the question is about a single-month breakdown (e.g., Opex breakdown for June), set 'month'. "
    "If not given, leave fields null. Do not include any fields not in the schema."
)

class CFOAgent:
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def classify(self, question: str) -> dict:
        resp = self.client.chat.completions.create(
            model = self.model,
            messages=[
                {"role": "system", "content": SYSTEM_GUIDE},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_schema", "json_schema": INTENT_SCHEMA},
            temperature=0,
        )

        msg = resp.choices[0].message
        if hasattr(msg, "parsed") and msg.parsed is not None:
            return msg.parsed
        else:
            return json.loads(msg.content)

    def narrate(self, question: str, intent: dict, df_preview: dict) -> str:
        prompt = (
            "You are a CFO analyst. Write a crisp, non-hype summary (1–3 sentences) for a board slide. "
            "Use USD (no decimals on dollars), show % with one decimal if applicable, and mention the period(s). "
            "Do not invent numbers. Only describe the JSON provided."
        )
        payload = json.dumps({"intent": intent, "data_preview": df_preview})

        resp = self.client.responses.create(
            model=self.model, 
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": f"Question: {question}"}]},
                {"role": "user", "content": [{"type": "input_text", "text": payload}]},
            ],
            temperature=0,
        )

        return resp.output_text