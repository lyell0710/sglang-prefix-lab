#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

utc_stamp() {
  date -u +%Y%m%dT%H%M%S
}

iso_utc() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

sglang_sha() {
  local upstream_dir=${SGLLAB_UPSTREAM_DIR:-/root/repos/sglang-v0.5.18}
  git -C "$upstream_dir" rev-parse HEAD 2>/dev/null || printf 'wheel-v0.5.18'
}

evidence_sha() {
  git -C "$project_root" rev-parse HEAD 2>/dev/null || printf 'pre-commit'
}

gpu_name() {
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort -u | paste -sd+ - || printf 'unavailable'
}

driver_version() {
  nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || printf 'unavailable'
}

prov_line() {
  local command_text=${1:?usage: prov_line '<complete command>' [seed]}
  local seed=${2:-none}
  printf '# provenance: env=%s sha=%s evidence_sha=%s cmd="%s" date=%s gpu="%s" driver=%s seed=%s\n' \
    "${SGLLAB_ENV_NAME:-sglang-lab}" "$(sglang_sha)" "$(evidence_sha)" "$command_text" "$(iso_utc)" \
    "$(gpu_name)" "$(driver_version)" "$seed"
}

