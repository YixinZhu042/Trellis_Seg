#!/usr/bin/env bash
# TAIJI appends <HOST_NUM> <NODE_RANK> <MASTER_ADDR>; MASTER_ADDR is unused.
set -euo pipefail
if [[ $# -ne 3 ]]; then echo 'Usage: bash scripts/prepare_dit_supervision_taiji.sh HOST_NUM NODE_RANK MASTER_ADDR' >&2; exit 2; fi
SEG_PREP_NODES=$1
SEG_PREP_NODE_RANK=$2
[[ "$SEG_PREP_NODES" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid HOST_NUM' >&2; exit 2; }
[[ "$SEG_PREP_NODE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] && (( SEG_PREP_NODE_RANK < SEG_PREP_NODES )) || { echo 'Invalid NODE_RANK' >&2; exit 2; }
SEG_PREP_PROJECT_DIR=${SEG_PREP_PROJECT_DIR:-/path/to/Trellis_Seg}
SEG_PREP_PYTHON=${SEG_PREP_PYTHON:-/path/to/trellis-env/bin/python}
SEG_PREP_OUTPUT_DIR=${SEG_PREP_OUTPUT_DIR:-${SEG_PREP_PROJECT_DIR}/data/dit_decoded_full_v1}
SEG_PREP_GPUS_PER_NODE=${SEG_PREP_GPUS_PER_NODE:-8}
SEG_PREP_SELECT_ONLY=${SEG_PREP_SELECT_ONLY:-0}
SEG_PREP_MERGE_ONLY=${SEG_PREP_MERGE_ONLY:-0}
[[ "$SEG_PREP_GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid GPU count' >&2; exit 2; }
cd "$SEG_PREP_PROJECT_DIR"
mkdir -p "$SEG_PREP_OUTPUT_DIR/logs"
world=$(( SEG_PREP_NODES * SEG_PREP_GPUS_PER_NODE ))
selection_matches_current_contract() {
    "$SEG_PREP_PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("format_version")==2 and len(d["samples"])==22909 and d["split_counts"]=={"train":21885,"validation":512,"test":512} and len(d.get("adaptive_resample_candidates",[]))==29 and not d.get("excluded_over_max_tokens")' "$SEG_PREP_OUTPUT_DIR/selection.json"
}
if (( SEG_PREP_NODE_RANK == 0 )) && [[ ! -f "$SEG_PREP_OUTPUT_DIR/selection.json" ]]; then
    "$SEG_PREP_PYTHON" scripts/prepare_dit_supervision.py --select --use_all \
        --val_samples 512 --test_samples 512 --max_tokens 8192 --output_dir "$SEG_PREP_OUTPUT_DIR"
fi
for attempt in $(seq 1 600); do
    if [[ -f "$SEG_PREP_OUTPUT_DIR/selection.json" ]] && \
       selection_matches_current_contract >/dev/null 2>&1; then
        break
    fi
    sleep 2
done
selection_matches_current_contract || { echo 'selection.json does not match the full-data v1 contract' >&2; exit 1; }
if [[ "$SEG_PREP_SELECT_ONLY" == 1 ]]; then exit 0; fi
if [[ "$SEG_PREP_MERGE_ONLY" == 1 ]]; then
    if (( SEG_PREP_NODE_RANK == 0 )); then
        "$SEG_PREP_PYTHON" scripts/prepare_dit_supervision.py --merge --output_dir "$SEG_PREP_OUTPUT_DIR"
        "$SEG_PREP_PYTHON" scripts/audit_prepared_dit_supervision.py \
            --data_dir "$SEG_PREP_OUTPUT_DIR" --output "$SEG_PREP_OUTPUT_DIR/prepared_audit.json"
        exec "$SEG_PREP_PYTHON" scripts/prepare_dit_supervision.py --check --require_prepared \
            --output_dir "$SEG_PREP_OUTPUT_DIR"
    fi
    exit 0
fi
"$SEG_PREP_PYTHON" scripts/prepare_dit_supervision.py --check --check_workers 64 --output_dir "$SEG_PREP_OUTPUT_DIR"
pids=()
for local_rank in $(seq 0 $((SEG_PREP_GPUS_PER_NODE-1))); do
    global_rank=$(( SEG_PREP_NODE_RANK * SEG_PREP_GPUS_PER_NODE + local_rank ))
    CUDA_VISIBLE_DEVICES=$local_rank "$SEG_PREP_PYTHON" scripts/prepare_dit_supervision.py \
        --output_dir "$SEG_PREP_OUTPUT_DIR" --rank "$global_rank" --world_size "$world" --device cuda:0 \
        --max_tokens 8192 --fail_on_error \
        > "$SEG_PREP_OUTPUT_DIR/logs/rank${global_rank}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
