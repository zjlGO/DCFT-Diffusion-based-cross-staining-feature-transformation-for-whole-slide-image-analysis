#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Use the active Python environment by default. To select the author's local
# environment explicitly, run with CONDA_ENV=AI4M.
if [[ -n "${CONDA_ENV:-}" && "${SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "CONDA_ENV was set but conda is unavailable." >&2
    exit 1
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  python -m featstaindiff.train --help
  exit 0
fi

: "${HE_FEATURES:?Set HE_FEATURES to the paired H&E feature file}"
: "${IHC_FEATURES:?Set IHC_FEATURES to the paired IHC feature file}"

args=(
  --he-features "$HE_FEATURES"
  --ihc-features "$IHC_FEATURES"
  --output-dir "${OUTPUT_DIR:-runs/default}"
)
if [[ "${ASSUME_ALIGNED_ORDER:-0}" == "1" ]]; then
  args+=(--assume-aligned-order)
fi
python -m featstaindiff.train "${args[@]}" "$@"
