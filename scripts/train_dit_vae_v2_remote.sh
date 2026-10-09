#!/usr/bin/env bash
# Run once on the controller. Remote API manages three guarded persistent sessions.
set -euo pipefail
DIT_V2_PROJECT="${TRELLIS_SEG_PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$DIT_V2_PROJECT"
args=(--config configs/dit_vae_v2_parts64.json --phase train)
case "${1:-status}" in
    status) args+=(--status) ;;
    launch) args+=(--launch) ;;
    resume) args+=(--launch --resume) ;;
    dry-run) ;;
    *) echo 'Usage: train_dit_vae_v2_remote.sh [status|launch|resume|dry-run]' >&2; exit 2 ;;
esac
exec python3 -B scripts/dit_vae_v2_remote.py "${args[@]}"
