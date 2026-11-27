#!/usr/bin/env bash
# Run CPU input-contract checks in Ubuntu-created environments; no dataset needed.
set -euo pipefail

if [[ "$(uname -s)" != Linux ]]; then
  echo 'Ubuntu/Linux 터미널에서 실행하세요.' >&2
  exit 2
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$PROJECT_ROOT"
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

models=(hvq_trans dinomaly_small u_cat rtdetrv2_s yolo11s ecf_yolov7_tiny)
if [[ $# -gt 1 ]]; then
  echo '사용법: bash project/scripts/verify_ubuntu.sh [모델 식별자]' >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  found=false
  for model in "${models[@]}"; do
    [[ "$model" != "$1" ]] || found=true
  done
  if [[ "$found" != true ]]; then
    echo "알 수 없는 모델: $1" >&2
    exit 2
  fi
  models=("$1")
fi

for model in "${models[@]}"; do
  interpreter="$PROJECT_ROOT/.venvs/$model/bin/python"
  if [[ ! -x "$interpreter" ]]; then
    echo "$model: Ubuntu 환경이 없습니다. Python 3.10으로 setup_envs.py를 먼저 실행하세요." >&2
    exit 2
  fi
  report="$PROJECT_ROOT/workspace/artifacts/ubuntu/$model"
  mkdir -p -- "$report"
  printf '\n=== %s: 환경 검사 ===\n' "$model"
  "$interpreter" -m relearn.cli doctor | tee "$report/doctor.json"
  "$interpreter" -m pip check 2>&1 | tee "$report/pip_check.log"
  printf '\n=== %s: 합성 CSV 로더 검사 ===\n' "$model"
  "$interpreter" -m unittest discover -s "$PROJECT_ROOT/tests" -v 2>&1 | tee "$report/tests.log"
done
echo '선택한 Ubuntu 환경의 입력 검사가 모두 통과했습니다. 모델 학습 검증은 별도입니다.'
