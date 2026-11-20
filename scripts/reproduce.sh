#!/usr/bin/env bash
# Reproduce the complete experiment from a fresh clone.
#   bash scripts/reproduce.sh            # full run (193 images × 41 conditions)
#   bash scripts/reproduce.sh --quick    # ~20 images, 4 condition families (a few minutes)
set -euo pipefail
cd "$(dirname "$0")/.."

QUICK_ARGS=()
if [[ "${1:-}" == "--quick" ]]; then
  QUICK_ARGS=(--max-images 20 --conditions gaussian_blur underexposure enhance__clahe enhance__sharpen)
fi

uv sync
uv run skin-research doctor
uv run skin-research download-data                    # 53 MB, ISIC 2018 Task 3 Validation
uv run skin-research prepare-data --fetch-lesion-ids
uv run skin-research validate-data
uv run skin-research download-model                   # 16 MB, pinned revision
uv run skin-research demo --image-id ISIC_0034329
uv run skin-research experiment "${QUICK_ARGS[@]}"
uv run pytest -q

echo
echo "Report: outputs/reports/isic2018_val_quality/report.md"
