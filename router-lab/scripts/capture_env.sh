#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$project_root/scripts/provenance.sh"

raw_dir=${1:?usage: capture_env.sh <data/raw/EXP-Sxx>}
python_bin=${SGLLAB_PYTHON:-/root/venvs/sglang-lab/bin/python}
prefix=$(utc_stamp)
mkdir -p "$raw_dir"

summary_path="$raw_dir/${prefix}_env_summary.txt"
freeze_path="$raw_dir/${prefix}_pip_freeze.txt"
check_path="$raw_dir/${prefix}_dependency_check.txt"

{
  prov_line "bash scripts/capture_env.sh $raw_dir"
  printf 'python_executable=%s\n' "$python_bin"
  printf 'upstream_dir=%s\n' "${SGLLAB_UPSTREAM_DIR:-/root/repos/sglang-v0.5.18}"
  printf 'upstream_sha=%s\n' "$(sglang_sha)"
  printf 'upstream_describe=%s\n' "$(git -C "${SGLLAB_UPSTREAM_DIR:-/root/repos/sglang-v0.5.18}" describe --tags --always)"
  printf 'model_smoke_snapshot=%s\n' \
    '/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'
  printf 'model_benchmark_snapshot=%s\n' \
    '/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218'
  "$python_bin" - <<'PY'
import importlib.metadata as metadata
import platform
import torch

packages = (
    "sglang",
    "sglang-router",
    "torch",
    "triton",
    "transformers",
    "flashinfer-python",
    "sglang-kernel",
)
print(f"python={platform.python_version()}")
for package in packages:
    print(f"{package}={metadata.version(package)}")
print(f"torch_version_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"device_count={torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    print(
        f"gpu_{index}={torch.cuda.get_device_name(index)} "
        f"capability={torch.cuda.get_device_capability(index)}"
    )
PY
  printf 'nvcc='; nvcc --version | sed -n 's/.*release \([^,]*\).*/\1/p'
  printf 'driver='; nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u | paste -sd+ -
} > "$summary_path"

{
  prov_line "uv pip freeze --python $python_bin"
  uv pip freeze --python "$python_bin"
} > "$freeze_path"

{
  prov_line "uv pip check --python $python_bin"
  uv pip check --python "$python_bin"
} > "$check_path" 2>&1

printf '%s\n%s\n%s\n' "$summary_path" "$freeze_path" "$check_path"

