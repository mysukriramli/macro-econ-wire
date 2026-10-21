#!/usr/bin/env python3
"""
Autonomous Global Macroeconomic Wire
Ingests raw economic indicators from the US Federal Reserve, calculates 
deltas, and synthesizes machine-readable trading directives for AI agents.
"""

import os
import json
import logging
from datetime import datetime, timezone
import requests
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

FRED_API_KEY = os.getenv("FRED_API_KEY")
DATA_DIR = "data"
ARCHIVE_DIR = os.path.join(DATA_DIR, "archive")
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# The core economic indicators agents need for macro reasoning
INDICATORS = {
    "CPIAUCSL": "US Consumer Price Index (Inflation)",
    "FEDFUNDS": "Effective Federal Funds Rate",
    "UNRATE": "US Unemployment Rate",
    "T10Y2Y": "10-Year Minus 2-Year Treasury Yield Spread"
}

class MacroIndicator(BaseModel):
    symbol: str
    name: str
    latest_value: float
    latest_date: str
    previous_value: float
    delta: float
    agent_directive: str


def fetch_fred_series(series_id: str) -> dict:
    """Fetches the two most recent data points for a specific FRED series."""
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
        f"&sort_order=desc&limit=2"
    )
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        obs = res.json().get("observations", [])
        if len(obs) >= 2:
            return {
                "latest_val": float(obs[0]["value"]),
                "latest_date": obs[0]["date"],
                "prev_val": float(obs[1]["value"])
            }
    except Exception as e:
        logging.error(f"Failed to fetch {series_id}: {e}")
    return {}


def synthesize_agent_directive(symbol: str, name: str, latest: float, delta: float) -> str:
    """Uses a low-cost LLM to translate raw math into an operational prompt directive."""
    api_key = os.getenv("LLM_API_KEY")
    api_base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    if not api_key:
        return "Monitor historical baselines."

    system_prompt = (
        "You are an elite quantitative strategist algorithm. Given a macroeconomic indicator and its recent shift, "
        "write a single, highly dense sentence instructing an autonomous trading agent how to adjust its risk parameters or portfolio weighting. "
        "Be extremely direct. No fluff."
    )
    user_prompt = f"Indicator: {name} ({symbol})\nLatest Value: {latest}\nChange from last period: {delta}"

    try:
        res = requests.post(
            f"{api_base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            },
            timeout=15
        )
        return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.error(f"LLM directive generation failed for {symbol}: {e}")
        return "Execute standard risk weighting."


def run_ingestion():
    if not FRED_API_KEY:
        logging.error("FRED_API_KEY is missing. Halting execution.")
        return

    records = []
    md_lines = [
        "# Global Macroeconomic & Yield Radar",
        "> Ground-truth economic telemetry and derived trading directives for autonomous financial agents.\n",
        "## Active Economic Baselines"
    ]

    for symbol, name in INDICATORS.items():
        data = fetch_fred_series(symbol)
        if not data:
            continue

        latest = data["latest_val"]
        prev = data["prev_val"]
        delta = round(latest - prev, 3)

        directive = synthesize_agent_directive(symbol, name, latest, delta)

        record = MacroIndicator(
            symbol=symbol,
            name=name,
            latest_value=latest,
            latest_date=data["latest_date"],
            previous_value=prev,
            delta=delta,
            agent_directive=directive
        )
        records.append(record)

        trend = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        md_lines.extend([
            f"### {name} (`{symbol}`)",
            f"- **Current Rate:** {latest} (Date: {data['latest_date']})",
            f"- **Delta:** {delta} {trend}",
            f"- **Agent Directive:** {directive}\n"
        ])

    if not records:
        return

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    
    payload = {
        "metadata": {
            "generated_at": now.isoformat(),
            "source": "US Federal Reserve (FRED)",
            "schema_version": "1.0.0"
        },
        "indicators": [r.model_dump() for r in records]
    }

    # Write rolling active endpoints
    with open(os.path.join(DATA_DIR, "today.json"), "w") as f:
        json.dump(payload, f, indent=2)
    
    with open(os.path.join(DATA_DIR, "llms.txt"), "w") as f:
        f.write("\n".join(md_lines))

    # Write permanent historical log
    month_dir = os.path.join(ARCHIVE_DIR, now.strftime("%Y"), now.strftime("%m"))
    os.makedirs(month_dir, exist_ok=True)
    with open(os.path.join(month_dir, f"{date_str}.json"), "w") as f:
        json.dump(payload, f, indent=2)

    logging.info(f"Macro Wire updated with {len(records)} indicators.")


if __name__ == "__main__":
    run_ingestion()
