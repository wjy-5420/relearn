"""Fetch pinned official sources into ignored third_party; never modify upstream history."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="all")
    args = parser.parse_args()
    models = json.loads((ROOT / "project/configs/models.json").read_text(encoding="utf-8"))
    if args.model != "all" and args.model not in models:
        parser.error("알 수 없는 모델")
    for name, spec in models.items():
        if args.model not in ("all", name):
            continue
        if not spec["upstream"] or not spec["commit"] or spec["commit"].startswith("pypi:"):
            print(f"SKIP {name}: 공식 소스 미확인 또는 PyPI 설치 대상")
            continue
        target = ROOT / "workspace/third_party" / name
        if not target.exists():
            subprocess.run(["git", "clone", spec["upstream"], str(target)], check=True)
        status = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"], text=True)
        if status.strip():
            raise RuntimeError(f"사용자 수정이 있는 upstream은 건드리지 않습니다: {target}")
        current = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if current != spec["commit"]:
            subprocess.run(["git", "-C", str(target), "fetch", "origin", spec["commit"]], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "--detach", spec["commit"]], check=True)
        print(f"PINNED {name}: {spec['commit']}")


if __name__ == "__main__":
    main()
