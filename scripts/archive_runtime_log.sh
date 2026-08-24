#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$project_root/scripts/provenance.sh"

exp_id=${1:?usage: archive_runtime_log.sh <EXP-Sxx> <service> <label> <PASS|FAIL>}
service=${2:?}
label=${3:?}
run_status=${4:?}
runtime_log="$project_root/runtime/$service.log"
runtime_cmd="$project_root/runtime/$service.cmd"
[[ -f "$runtime_log" ]] || { printf 'missing runtime log: %s\n' "$runtime_log" >&2; exit 1; }

raw_dir="$project_root/data/raw/$exp_id"
prefix=$(utc_stamp)
output_path="$raw_dir/${prefix}_${service}_${label}.log"
mkdir -p "$raw_dir"

{
  prov_line "bash scripts/archive_runtime_log.sh $exp_id $service $label $run_status"
  printf '# run_status=%s service=%s\n' "$run_status" "$service"
  if [[ -f "$runtime_cmd" ]]; then
    printf '# launched_command='
    sed -n '1p' "$runtime_cmd"
  fi
  sed -n '1,$p' "$runtime_log"
} > "$output_path"

printf '%s\n' "$output_path"

