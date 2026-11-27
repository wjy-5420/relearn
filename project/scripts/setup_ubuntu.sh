#!/usr/bin/env bash
# Create native Ubuntu environments; never copy Windows environments.
set -euo pipefail

if [[ "$(uname -s)" != Linux ]]; then
  echo 'Ubuntu/Linux 터미널에서 실행하세요.' >&2
  exit 2
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$PROJECT_ROOT"
if [[ -n "${PYTHON310:-}" ]]; then
  interpreter="$PYTHON310"
elif [[ -x "$PROJECT_ROOT/workspace/runtime/python310/bin/python" ]]; then
  interpreter="$PROJECT_ROOT/workspace/runtime/python310/bin/python"
else
  interpreter=python3.10
fi
if ! command -v -- "$interpreter" >/dev/null 2>&1; then
  echo 'Python 3.10이 필요합니다. 설치된 Python 3.10 경로를 PYTHON310으로 지정할 수 있습니다.' >&2
  exit 2
fi
export PYTHONIOENCODING=utf-8
"$interpreter" "$PROJECT_ROOT/project/scripts/setup_envs.py" --install model "$@"
