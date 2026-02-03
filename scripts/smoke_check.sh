#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "venv python not found at $PYTHON" >&2
  exit 1
fi

"$PYTHON" - <<'PY'
import librosa, numba, llvmlite, mutagen, numpy, scipy, requests
print("ok:", {
    "librosa": librosa.__version__,
    "numba": numba.__version__,
    "llvmlite": llvmlite.__version__,
    "mutagen": mutagen.__version__,
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "requests": requests.__version__,
})
PY
