"""Verify installed environments with synthetic CSVs; no real data or GPU training."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
MODELS = json.loads((ROOT / "project/configs/models.json").read_text(encoding="utf-8"))


def verify(name):
    python = ROOT / ".venvs" / name / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    report = ROOT / "workspace/artifacts/environments" / name
    report.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = {"model": name, "installed": python.exists(), "verification": "synthetic_csv_adapter_only"}
    if python.exists():
        for key, args in (("pip_check", ["-m", "pip", "check"]),
                          ("tests", ["-m", "unittest", "discover", "-s", "tests", "-v"])):
            run = subprocess.run([str(python), *args], cwd=ROOT, env=environment,
                                 text=True, encoding="utf-8", capture_output=True, timeout=180)
            (report / (key + ".log")).write_text(run.stdout + run.stderr, encoding="utf-8")
            result[key] = {"passed": run.returncode == 0}
    (report / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(verify, MODELS))
    output = ROOT / "workspace/artifacts/environments/summary.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(output.read_text())
    return 0 if all(r.get("tests", {}).get("passed") and r.get("pip_check", {}).get("passed") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
