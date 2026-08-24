#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$project_root/scripts/provenance.sh"

service=${1:?usage: run_smoke.sh <service> <base_url> <model_name> [EXP-Sxx]}
base_url=${2:?}
model_name=${3:?}
exp_id=${4:-EXP-S01}
raw_dir="$project_root/data/raw/$exp_id"
prefix=$(utc_stamp)
mkdir -p "$raw_dir"

smoke_path="$raw_dir/${prefix}_${service}_smoke_response.txt"
metrics_path="$raw_dir/${prefix}_${service}_metrics.txt"
model_info_path="$raw_dir/${prefix}_${service}_model_info.txt"
gpu_path="$raw_dir/${prefix}_${service}_gpu_snapshot.txt"

set +e
{
  prov_line "bash scripts/run_smoke.sh $service $base_url $model_name $exp_id"
  "$project_root/scripts/smoke_client.py" --base-url "$base_url" --model "$model_name"
} > "$smoke_path" 2>&1
smoke_rc=$?
set -e

{
  prov_line "curl -fsS $base_url/metrics"
  curl -fsS "$base_url/metrics"
} > "$metrics_path"

{
  prov_line "curl -fsS $base_url/model_info"
  curl -fsS "$base_url/model_info"
  printf '\n'
} > "$model_info_path"

{
  prov_line "nvidia-smi smoke snapshot for $service"
  nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,pstate,temperature.gpu,power.draw,power.limit \
    --format=csv,noheader
  printf '\n[compute_processes]\n'
  nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader 2>/dev/null || true
} > "$gpu_path"

printf 'smoke_rc=%s\nsmoke=%s\nmetrics=%s\nmodel_info=%s\ngpu=%s\n' \
  "$smoke_rc" "$smoke_path" "$metrics_path" "$model_info_path" "$gpu_path"
exit "$smoke_rc"

