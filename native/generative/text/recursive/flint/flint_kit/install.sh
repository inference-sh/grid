#!/usr/bin/env bash
# inference.sh copy of the Flint kit's sglang/install.sh (RecursiveSI/flint@2a247be0), changed only
# where SGLang 0.5.20 forces it: between 0.5.16 and 0.5.20 SGLang moved the vendored FA4 kernel from
# jit_kernel/flash_attn/cute/ to kernels/ops/attention/flash_attn/cute/, so the kernel path and its
# hashes are 0.5.20's, and the kit's unchanged patch is applied to that file by name (it still names
# the old path). flint_sglang's model.py gets our 0.5.20 port (flint_sglang-sglang-0.5.20.patch).
# Every other pin, the FA4 patch and the wheel come from the kit as shipped.
#
# Install Flint support into an lmsysorg/sglang:v0.5.16-cu130 environment.
# Used by the Dockerfile; can also be run (and re-run, e.g. to upgrade) inside an existing
# container of that image, as root or not.
#
#   bash install.sh <dir containing fa4-learned-sink-fix.patch and one flint_sglang-*.whl>
#
# Needs network access to pypi.org/files.pythonhosted.org, flashinfer.ai and github.com
# (FlashInfer release assets). For air-gapped hosts, pre-download the wheels and run with
# PIP_NO_INDEX=1 PIP_FIND_LINKS=<wheel dir>.
set -euo pipefail

SRC_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# The image installs SGLang into the system Python; allow pip to update it there (PEP 668).
export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore PIP_NO_CACHE_DIR=1

# flashinfer 0.6.16.post1: the base image's 0.6.14 can hang trtllm MoE kernels on Blackwell.
# The python, cubin and jit-cache wheels must move together.
pip install --force-reinstall --no-deps \
  --extra-index-url "https://flashinfer.ai/whl" \
  --extra-index-url "https://flashinfer.ai/whl/cu130" \
  "flashinfer-python==0.6.16.post1" \
  "flashinfer-cubin==0.6.16.post1" \
  "flashinfer-jit-cache==0.6.16.post1"

# CuTe-DSL runtime used to JIT-compile the FA4 attention kernels (4.6.0/4.6.1 hang on sm_103).
pip install --force-reinstall --no-deps \
  "nvidia-cutlass-dsl==4.6.2" \
  "nvidia-cutlass-dsl-libs-base==4.6.2" \
  "nvidia-cutlass-dsl-libs-core==4.6.2" \
  "nvidia-cutlass-dsl-libs-cu12==4.6.2" \
  "nvidia-cutlass-dsl-libs-cu13==4.6.2" \
  "quack-kernels==0.6.4" \
  "apache-tvm-ffi==0.1.12"

python3 - <<'EOF'
from importlib.metadata import version as v
expected = {
    "flashinfer-python": "0.6.16.post1", "flashinfer-cubin": "0.6.16.post1", "flashinfer-jit-cache": "0.6.16.post1",
    "nvidia-cutlass-dsl": "4.6.2", "nvidia-cutlass-dsl-libs-base": "4.6.2", "nvidia-cutlass-dsl-libs-core": "4.6.2",
    "nvidia-cutlass-dsl-libs-cu12": "4.6.2", "nvidia-cutlass-dsl-libs-cu13": "4.6.2",
    "quack-kernels": "0.6.4", "apache-tvm-ffi": "0.1.12",
}
bad = {p: v(p) for p, want in expected.items() if not v(p).startswith(want)}
assert not bad, f"unexpected package versions: {bad}"
EOF

# FA4 learned-sink normalization and TMEM-allocation barrier fix for SGLang's vendored
# FlashAttention-4 SM100 forward kernel. Required for --attention-backend fa4.
SGL_DIR="$(python3 -c "import pathlib, sglang; print(pathlib.Path(sglang.__file__).parent)")"
KERNEL="$SGL_DIR/kernels/ops/attention/flash_attn/cute/flash_fwd_sm100.py"
PATCHED_SHA=9e2e4032f7d75a83c204c9f2523906290e918aef6ffa68305cf91370b081871a
if echo "$PATCHED_SHA  $KERNEL" | sha256sum -c --status -; then
  echo "FA4 learned-sink fix already applied."
else
  echo "37e40dbaae7d22c87210c4e0541790e9ace2d66e469259faf5b299f746403dfd  $KERNEL" | sha256sum -c -
  patch "$KERNEL" < "$SRC_DIR/fa4-learned-sink-fix.patch"
  echo "$PATCHED_SHA  $KERNEL" | sha256sum -c -
fi

shopt -s nullglob
wheels=("$SRC_DIR"/flint_sglang-*.whl)
if [[ ${#wheels[@]} -ne 1 ]]; then
  echo "expected exactly one flint_sglang-*.whl in $SRC_DIR, found ${#wheels[@]}" >&2
  exit 1
fi
pip install --force-reinstall --no-deps "${wheels[0]}"

# flint_sglang 0.1.4 targets SGLang 0.5.16. In 0.5.20 get_global_server_args() raises (resolved
# values are read from runtime_context), and the vendored FA4 kernel moved. Our patch, kept next to
# this script, ports both; it applies only to the wheel's model.py as shipped.
FLINT_MODEL="$(python3 -c "import pathlib, importlib.util; print(pathlib.Path(importlib.util.find_spec('flint_sglang').origin).parent / 'model.py')")"
echo "948d721be04e0af086021a1d26bd9a7620d2313bf09442739bebbebcf6cb9bf3  $FLINT_MODEL" | sha256sum -c -
patch "$FLINT_MODEL" < "$(dirname "${BASH_SOURCE[0]}")/flint_sglang-sglang-0.5.20.patch"
echo "f165d27d1a9047e692d9a4360bfe6e3b2836903781ea8cf80141a87604247d5a  $FLINT_MODEL" | sha256sum -c -
python3 -c "from importlib.metadata import entry_points; assert any(e.name == 'flint' for e in entry_points(group='sglang.srt.plugins'))"
echo "Flint support installed ($(basename "${wheels[0]}"))."
