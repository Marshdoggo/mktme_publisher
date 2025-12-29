from pathlib import Path
import subprocess
import os
import sys

BASE = Path(__file__).resolve().parent

# Try common venv layouts
candidates = [
    BASE / "venv" / "bin" / "python",
    BASE / "venv" / "bin" / "python3",
    BASE / ".venv" / "bin" / "python",
    BASE / ".venv" / "bin" / "python3",
]

PY = next((p for p in candidates if p.exists()), None)

if PY is None:
    print("[run_publish] Could not find venv Python in:")
    for c in candidates:
        print("   ", c)
    print()
    print("Create it with:")
    print("  python3 -m venv ~/mktme_publisher/venv")
    print("  source ~/mktme_publisher/venv/bin/activate")
    print("  pip install pandas numpy yfinance requests pyarrow python-dotenv")
    sys.exit(1)

env = os.environ.copy()  # .env will be loaded by publish_mktme.py
cmd = [str(PY), str(BASE / "publish_mktme.py")] + sys.argv[1:]

print("[run_publish] Using interpreter:", PY)
print("[run_publish] Running:", " ".join(cmd))
print()

print("[run_publish] ENV SUMMARY:")
for k in [
    "MKTME_APP_SRC",
    "MKTME_DATA_REPO",
    "PYTHONPATH",
    "MKTME_FORCE_REFRESH",
    "MKTME_MANIFEST_URL",
]:
    print(f"  {k} =", env.get(k))
print()

# Don't use check=True so we can see the real traceback from publish_mktme.py
result = subprocess.run(cmd, env=env)

# Validate expected outputs when publish succeeds.
# We keep these checks in the wrapper so failures are loud in CI.
data_repo = Path(env.get("MKTME_DATA_REPO", ""))
reports_dir = data_repo / "reports"
index_json = reports_dir / "index.json"

# We expect at minimum:
# - reports/index.json (global index)
# - a universe folder with at least one YYYY-MM-DD subfolder containing report.md (or report.json)
# The universe folder name can vary (sp500, fx, etc.), so we search.

def _find_any_report_leaf(root: Path) -> Path | None:
    """Return any file that proves at least one report exists.

    Supports both layouts:

      Old:
        reports/<universe>/<YYYY-MM-DD>/report.md
        reports/<universe>/<YYYY-MM-DD>/report.json

      New:
        reports/<universe>/<YYYY-MM-DD>.md
        reports/<universe>/<YYYY-MM-DD>.json
        reports/<universe>/<YYYY-MM-DD>.facts.json (optional)

    We intentionally avoid treating index/manifest files as a "report".
    """
    if not root.exists():
        return None

    # New flat layout (preferred)
    for p in root.glob("*/*.md"):
        # exclude non-report markdown if any appear later
        if p.name.lower() != "readme.md":
            return p
    for p in root.glob("*/*.json"):
        # exclude index.json / manifest.json / facts-only files if they happen to match
        if p.name in {"index.json", "manifest.json"}:
            continue
        return p

    # Old nested layout
    for p in root.glob("*/*/report.md"):
        return p
    for p in root.glob("*/*/report.json"):
        return p

    return None

leaf = _find_any_report_leaf(reports_dir)

if result.returncode == 0:
    missing = []
    if not index_json.exists():
        missing.append(str(index_json))
    if leaf is None:
        missing.append(
            str(
                reports_dir
                / "<universe>/<YYYY-MM-DD>.{md,json} OR <universe>/<YYYY-MM-DD>/report.{md,json}"
            )
        )

    if missing:
        print("[run_publish] ERROR: publish succeeded but expected report artifacts are missing:")
        for m in missing:
            print("  -", m)
        print("\n[run_publish] Debug listing of reports/:")
        try:
            if reports_dir.exists():
                for p in sorted(reports_dir.rglob("*") ):
                    if p.is_file():
                        print("  ", p.relative_to(data_repo))
            else:
                print("   reports/ directory does not exist")
        except Exception as e:
            print("   (listing failed)", e)
        sys.exit(2)

print()
print(f"[run_publish] publish_mktme.py exited with code {result.returncode}")
sys.exit(result.returncode)