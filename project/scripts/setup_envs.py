"""Create six isolated environments. CPU is for loader tests, not an 8GB training claim."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[2]
MODELS = json.loads((ROOT / "project/configs/models.json").read_text(encoding="utf-8"))


def run(command):
    subprocess.run([str(c) for c in command], check=True, cwd=ROOT)


def normalize_editable_path(env):
    # Python 3.10 reads .pth using its locale. ASCII source keeps Korean paths
    # valid both with Windows cp949 and when the caller enables -X utf8.
    site = env / ("Lib/site-packages" if sys.platform == "win32" else "lib/python3.10/site-packages")
    for link in site.glob("__editable__.relearn_realiad-*.pth"):
        link.write_text("import sys; sys.path.insert(0, " + ascii(str(ROOT / "src")) + ")\n", encoding="ascii")


def main():
    os.environ["PYTHONIOENCODING"] = "utf-8"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    parser.add_argument("--install", choices=["none", "core", "model"], default="core")
    parser.add_argument("--torch-index", default="https://download.pytorch.org/whl/cpu")
    parser.add_argument("--reinstall-torch", action="store_true", help="CPU/CUDA wheel 전환 시 명시적으로 재설치")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        parser.error("레거시 모델 호환성을 위해 Python 3.10으로 실행하세요")
    for name in MODELS if args.model == "all" else [args.model]:
        spec = MODELS[name]
        env = ROOT / ".venvs" / name
        python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        foreign_python = env / ("bin/python" if sys.platform == "win32" else "Scripts/python.exe")
        if foreign_python.exists():
            parser.error(f"다른 OS에서 만든 환경을 재사용할 수 없습니다: {env}. OS별 프로젝트 폴더에서 환경을 새로 만드세요")
        if not python.exists():
            venv.EnvBuilder(with_pip=True).create(env)
        if args.install != "none":
            run([python, "-m", "pip", "install", "--disable-pip-version-check", "-r", ROOT / "project/environments/common.txt"])
            # Install Torch explicitly rather than letting a detector silently upgrade another model.
            torch_command = [python, "-m", "pip", "install", f"torch=={spec['torch']}",
                             f"torchvision=={spec['torchvision']}", "--index-url", args.torch_index]
            if args.reinstall_torch:
                installed_torch = subprocess.run([str(python), "-m", "pip", "show", "torch"], capture_output=True)
                if installed_torch.returncode:
                    parser.error("--reinstall-torch는 기존 Torch 환경을 전환할 때 사용하세요. 먼저 일반 설치가 필요합니다")
                torch_command.extend(["--force-reinstall", "--no-deps"])
            run(torch_command)
            if args.install == "model":
                run([python, "-m", "pip", "install", "-r", ROOT / spec["requirements"]])
            run([python, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", ROOT])
            normalize_editable_path(env)
            run([python, "-m", "pip", "check"])
            packages = json.loads(subprocess.check_output([str(python), "-m", "pip", "list", "--format=json"], text=True, encoding="utf-8"))
            freeze = "\n".join(f"{p['name']}=={p['version']}" for p in packages if p["name"] != "relearn-realiad") + "\n-e .\n"
            report_dir = ROOT / "workspace/artifacts/environments" / name
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "freeze.txt").write_text(freeze, encoding="utf-8")
            result = subprocess.check_output([str(python), "-m", "relearn.cli", "doctor"], text=True, encoding="utf-8")
            (report_dir / "doctor.json").write_text(result, encoding="utf-8")
        print(f"READY {name}: {python}", flush=True)


if __name__ == "__main__":
    main()
