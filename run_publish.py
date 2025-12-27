# ~/mktme_publisher/run_publish.py

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
cmd = [str(PY), str(BASE / "publish_mktme.py")]

print("[run_publish] Using interpreter:", PY)
print("[run_publish] Running:", " ".join(cmd))
print()

print("[run_publish] ENV SUMMARY:")
for k in [
    "MKTME_APP_SRC",
    "MKTME_DATA_REPO",
    "PYTHONPATH",
    "MKTME_FORCE_REFRESH",
]:
    print(f"  {k} =", env.get(k))
print()

# Don't use check=True so we can see the real traceback from publish_mktme.py
result = subprocess.run(cmd, env=env)

expected = Path(env.get("MKTME_DATA_REPO", "")) / "reports" / "index.json"
if result.returncode == 0 and not expected.exists():
    print("[run_publish] ERROR: publish succeeded but reports/index.json not found")
    sys.exit(2)

print()
print(f"[run_publish] publish_mktme.py exited with code {result.returncode}")
sys.exit(result.returncode)