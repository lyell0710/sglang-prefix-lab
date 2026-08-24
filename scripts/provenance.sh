#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
iso_utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
utc_stamp() { date -u +%Y%m%dT%H%M%S; }
prov_line() {
  local cmd_text=${1:?usage: prov_line '<cmd>' [seed]}; local seed=${2:-none}
  printf '# provenance: env=sglang-lab sha=%s evidence_sha=%s cmd="%s" date=%s gpu="%s" driver=%s seed=%s\n' \
    "$(git -C /root/repos/sglang-v0.5.18 rev-parse --short HEAD 2>/dev/null || echo wheel-v0.5.18)" \
    "$(git -C "$project_root" rev-parse --short HEAD 2>/dev/null || echo pre-commit)" \
    "$cmd_text" "$(iso_utc)" \
    "$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | paste -sd+ -)" \
    "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" "$seed"
}
