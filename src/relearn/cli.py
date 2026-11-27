import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

from .adapters import MODEL_NAMES
from .data import Manifest, load_config


def summary(value):
    if hasattr(value, "shape"):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {k: summary(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [summary(v) for v in value]
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description="Real-IAD CSV 기반 6모델 로더 (추출·분할 생성 없음)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models")
    sub.add_parser("doctor")
    for name in ("validate-csv", "smoke"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", required=True)
        cmd.add_argument("--report")
        if name == "validate-csv":
            cmd.add_argument("--check-files", action="store_true")
        else:
            cmd.add_argument("--model", choices=MODEL_NAMES, required=True)
            cmd.add_argument("--split", choices=("train", "val", "test"), default="train")
            cmd.add_argument("--batch-size", type=int, default=1)
            cmd.add_argument("--workers", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        code = 0
        if args.command == "models":
            result = json.loads((Path(__file__).resolve().parents[2] / "project/configs/models.json").read_text(encoding="utf-8"))
        elif args.command == "doctor":
            result = {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}
            for package in ("numpy", "Pillow", "torch", "torchvision"):
                try:
                    result[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    result[package] = "not installed"
            if result["torch"] != "not installed":
                import torch
                result["cuda_available"] = torch.cuda.is_available()
                result["cuda_build"] = torch.version.cuda
        else:
            manifest = Manifest.read(load_config(args.config))
            if args.command == "validate-csv":
                result = manifest.validate(args.check_files)
                code = 0 if result["valid"] else 2
            else:
                from .adapters import build_loader
                loader = build_loader(args.model, manifest, args.split, args.batch_size, args.workers)
                result = {"model": args.model, "split": args.split, "samples": len(loader.dataset),
                          "csv_sha256": manifest.digest, "batch": summary(next(iter(loader))),
                          "verification": "input_batch_only_not_model_training"}
        output = json.dumps(result, ensure_ascii=False, indent=2)
        if getattr(args, "report", None):
            path = Path(args.report)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(output + "\n", encoding="utf-8")
        print(output)
        return code
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
