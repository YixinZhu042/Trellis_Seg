#!/usr/bin/env bash
# B: fork best regular weights, synchronized label augmentation, small-part optimization.
set -euo pipefail
DIT_B_PROJECT="${TRELLIS_SEG_PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$DIT_B_PROJECT"
args=(--config configs/dit_vae_v2_parts64_b_smallparts.json --phase train)
case "${1:-status}" in
    status) args+=(--status) ;;
    launch) args+=(--launch) ;;
    resume) args+=(--launch --resume) ;;
    dry-run) ;;
    *) echo 'Usage: train_dit_b_smallparts_remote.sh [status|launch|resume|dry-run]' >&2; exit 2 ;;
esac
exec python3 -B scripts/dit_vae_v2_remote.py "${args[@]}"
