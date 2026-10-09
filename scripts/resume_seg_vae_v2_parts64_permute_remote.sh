#!/usr/bin/env bash
# Execute once on the controller; Remote API dispatches all three nodes.
set -euo pipefail
VAE_PERM_PROJECT="${TRELLIS_SEG_PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
VAE_PERM_SOURCE="${TRELLIS_VAE_RESUME_SOURCE:-${TRELLISSEG_STORAGE_ROOT:-/path/to/trellisseg_run}/checkpoints/seg_vae_v2_parts64/trainer_latest.pt}"
cd "$VAE_PERM_PROJECT"
args=(--config configs/seg_vae_v2_parts64_permute.json --manifest-workers 4)
case "${1:-launch}" in
    launch) args+=(--launch --resume-from "$VAE_PERM_SOURCE") ;;
    dry-run) args+=(--resume-from "$VAE_PERM_SOURCE") ;;
    status) args+=(--status) ;;
    resume) args+=(--launch --resume) ;;
    *) echo 'Usage: resume_seg_vae_v2_parts64_permute_remote.sh [launch|dry-run|status|resume]' >&2; exit 2 ;;
esac
exec python3 -B scripts/train_seg_vae_v2_remote.py "${args[@]}"
