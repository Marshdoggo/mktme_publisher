# ~/mktme_publisher/publish_mktme.py

import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

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