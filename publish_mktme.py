# ~/mktme_publisher/publish_mktme.py

import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import math
import numpy as np

from dotenv import load_dotenv
import pandas as pd

# ---- Load .env and basic config ----

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")

APP_SRC = Path(os.environ["MKTME_APP_SRC"])
DATA_REPO = Path(os.environ["MKTME_DATA_REPO"])
GITHUB_USER = os.environ.get("MKTME_GITHUB_USER", "marshdoggo")

UNIVERSES = ["sp500"]
LOOKBACK = 252  # trading days

print(f"[publisher] APP_SRC     = {APP_SRC}")
print(f"[publisher] DATA_REPO  = {DATA_REPO}")
print(f"[publisher] GITHUB_USER = {GITHUB_USER}")

# Make sure repo exists
DATA_REPO.mkdir(parents=True, exist_ok=True)

# Put your app's src on sys.path so imports work
if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))

from fetch_data import download_prices, download_prices_fx_window

# Report builders from the app repo
from ai_report import generate_daily_report, save_report_json, save_report_markdown

try:
    from ai_context import get_openai_client, build_view_context_text
except Exception:
    get_openai_client = None
    build_view_context_text = None


# ---- Helpers ----

def write_parquet(df: pd.DataFrame, universe: str) -> str:
    """Save a prices DataFrame and return relative path used in manifest."""
    out_dir = DATA_REPO / universe
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prices.parquet"
    df.to_parquet(out_path, engine="pyarrow")
    rel_path = f"{universe}/prices.parquet"
    print(f"[publisher] Wrote {universe}: {out_path}")
    return rel_path


ROOT = APP_SRC.parent

def load_tickers(universe: str) -> List[str]:
    """Load ticker symbols for a universe from the local cache_universes CSV."""
    cache_dir = ROOT / "data" / "cache_universes"
    name_map = {
        "sp500": "sp500.csv",
        "nasdaq100": "nasdaq100.csv",
        "dow30": "dow30.csv",
        # "fx": "fx.csv",  # enable later when we have a cached file
    }
    if universe not in name_map:
        raise ValueError(f"Unknown universe for load_tickers: {universe}")

    path = cache_dir / name_map[universe]
    if not path.exists():
        raise FileNotFoundError(f"Universe CSV not found: {path}")

    df = pd.read_csv(path)
    # Use the first column as symbols to avoid hard-coding a name
    if df.shape[1] == 0:
        raise ValueError(f"Universe CSV has no columns: {path}")
    col = df.columns[0]
    symbols = df[col].astype(str).dropna().unique().tolist()
    return symbols


def _to_naive_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx
    return df


def _compute_metrics_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute a compact metrics table from prices (close levels)."""
    prices = _to_naive_dt_index(prices)
    prices = prices.sort_index()

    rets = prices.pct_change().dropna(how="all")

    # Core stats
    mu = rets.mean(skipna=True)
    sig = rets.std(skipna=True, ddof=1)
    ann_sharpe = (mu / sig) * math.sqrt(252)

    # Sortino (downside deviation)
    downside = rets.where(rets < 0, 0)
    down_sig = downside.std(skipna=True, ddof=1)
    sortino = (mu / down_sig) * math.sqrt(252)

    # CAGR
    first = prices.iloc[0]
    last = prices.iloc[-1]
    n_days = (prices.index[-1] - prices.index[0]).days
    years = max(n_days / 365.25, 1e-9)
    cagr = (last / first) ** (1 / years) - 1

    # Max drawdown
    dd_vals = {}
    for col in prices.columns:
        s = prices[col].dropna()
        if s.empty:
            dd_vals[col] = np.nan
            continue
        roll_max = s.cummax()
        dd = (s / roll_max) - 1.0
        dd_vals[col] = float(dd.min())
    max_dd = pd.Series(dd_vals)

    # RSI(14)
    rsi_vals = {}
    window = 14
    for col in prices.columns:
        s = prices[col].dropna()
        if len(s) < window + 1:
            rsi_vals[col] = np.nan
            continue
        delta = s.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_vals[col] = float(rsi.iloc[-1])
    rsi14 = pd.Series(rsi_vals)

    out = pd.DataFrame({
        "Ticker": mu.index.astype(str),
        "Mean Daily Return": mu.values,
        "Daily Volatility (Std)": sig.values,
        "Annualized Sharpe": ann_sharpe.values,
        "Sortino Ratio": sortino.values,
        "CAGR": cagr.reindex(mu.index).values,
        "Max Drawdown": max_dd.reindex(mu.index).values,
        "RSI(14)": rsi14.reindex(mu.index).values,
    })

    # Stable ordering
    out["Ticker_upper"] = out["Ticker"].astype(str).str.upper()
    return out


def _write_report_artifacts(
    universe: str,
    prices: pd.DataFrame,
    lookback: int,
    github_user: str,
    data_repo: Path,
) -> None:
    """Create JSON + MD report files under mktme-data/reports/<universe>/ and update index.json."""
    prices = prices.copy()
    prices = _to_naive_dt_index(prices)

    asof_date = pd.to_datetime(prices.index.max()).date().isoformat()

    metrics_df = _compute_metrics_from_prices(prices)

    # Deterministic base report (tables, sector summary if available)
    daily = generate_daily_report(
        metrics_df=metrics_df,
        universe=universe,
        asof_date=asof_date,
        lookback=lookback,
        primary_rank_metric="Annualized Sharpe",
        top_n=5,
    )

    reports_dir = data_repo / "reports" / universe
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_path = reports_dir / f"{asof_date}.json"
    md_path = reports_dir / f"{asof_date}.md"

    save_report_json(daily.report, json_path)

    # If OpenAI is available + key is present, generate a narrative markdown brief.
    narrative_md = None
    if get_openai_client is not None and build_view_context_text is not None:
        try:
            client = get_openai_client(None)  # use env var in Actions
        except Exception:
            client = None

        if client is not None:
            try:
                # Build a compact context. Choose a sensible default lens.
                context_text = build_view_context_text(
                    metrics_df=metrics_df,
                    prices=prices,
                    x_metric="Daily Volatility (Std)",
                    y_metric="Annualized Sharpe",
                    universe=universe,
                    asof_ts=pd.to_datetime(prices.index.max()),
                    lookback=lookback,
                    highlight_upper=None,
                    top_table_rows=5,
                    include_cluster_summary=False,
                )

                instructions = (
                    "You are writing a concise daily market brief for a quantitative dashboard. "
                    "Use ONLY the provided context. Do not invent tickers or numbers. "
                    "Write in Markdown. Include: (1) 3–6 bullet headline takeaways; "
                    "(2) a short paragraph on the overall distribution (risk/return dispersion); "
                    "(3) a section called 'Leaders' referencing the top table rows by name; "
                    "(4) a short 'What to watch' list. "
                    "Do not exceed ~350 words."
                )

                resp = client.responses.create(
                    model="gpt-4.1-mini",
                    instructions=instructions,
                    input=f"CONTEXT:\n{context_text}\n\nBASE_REPORT_JSON:\n{json.dumps(daily.report)[:12000]}\n",
                )
                narrative_md = getattr(resp, "output_text", None)
            except Exception:
                narrative_md = None

    # Fallback to deterministic markdown if no narrative
    save_report_markdown(narrative_md or daily.markdown, md_path)

    # Update an index.json listing recent reports
    index_path = data_repo / "reports" / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    index = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            index = {}

    index.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
    index.setdefault("base_url", f"https://raw.githubusercontent.com/{github_user}/mktme-data/main")
    index.setdefault("universes", {})
    index["universes"].setdefault(universe, [])

    # Prepend newest, keep unique, cap to 14
    rel_json = f"reports/{universe}/{asof_date}.json"
    rel_md = f"reports/{universe}/{asof_date}.md"
    entry = {"date": asof_date, "json": rel_json, "md": rel_md}

    existing = index["universes"][universe]
    existing = [e for e in existing if e.get("date") != asof_date]
    index["universes"][universe] = [entry] + existing
    index["universes"][universe] = index["universes"][universe][:14]

    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    print(f"[publisher] Wrote report JSON: {json_path}")
    print(f"[publisher] Wrote report MD:   {md_path}")
    print(f"[publisher] Updated index:     {index_path}")


# ---- Main logic ----

def main() -> None:
    manifest: dict[str, str] = {}
    for universe in UNIVERSES:
        print(f"[publisher] Processing universe: {universe}")

        tickers = load_tickers(universe)
        print(f"[publisher]  tickers: {len(tickers)}")

        if universe == "fx":
            # Pass LOOKBACK as a positional arg
            prices = download_prices_fx_window(tickers, LOOKBACK)
        else:
            # Same here – no keyword
            prices = download_prices(tickers, LOOKBACK)

        rel = write_parquet(prices, universe)
        manifest[universe] = rel

        # Also publish daily report artifacts (JSON + Markdown)
        try:
            _write_report_artifacts(
                universe=universe,
                prices=prices,
                lookback=LOOKBACK,
                github_user=GITHUB_USER,
                data_repo=DATA_REPO,
            )
        except Exception as e:
            print(f"[publisher] Report generation failed for {universe}: {e}")

    # Build HTTP URLs for Streamlit to use
    base_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/mktme-data/main"
    http_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "universes": {
            u: {"parquet_url": f"{base_url}/{rel}"}
            for u, rel in manifest.items()
        },
    }

    manifest_path = DATA_REPO / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(http_manifest, f, indent=2)

    print(f"[publisher] Wrote manifest: {manifest_path}")
    print("[publisher] Done. Commit & push from mktme-data when ready.")


if __name__ == "__main__":
    main()