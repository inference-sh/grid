#!/usr/bin/env bash
# inference.sh copy of the Flint kit's sglang/serve_b300.sh (RecursiveSI/flint@2a247be0), changed only
# where SGLang 0.5.20 forces it: --disable-piecewise-cuda-graph was a deprecated alias of
# --cuda-graph-backend-prefill=disabled in 0.5.16 and is gone in 0.5.20, so the script passes the
# flag it stood for. serve_h200.sh needs no change and is run from the kit as shipped.
#
# One Flint replica on one Blackwell GPU. Validated on B300 (x86_64); B200 and GB200/GB300
# (arm64 image) use the same flags but are not yet validated.
#
#   MODEL=/path/to/flint CUDA_VISIBLE_DEVICES=0 PORT=30000 bash serve_b300.sh
#
# Extra arguments are passed through to sglang.launch_server.
set -euo pipefail

MODEL="${MODEL:-RecursiveSI/flint}"
PORT="${PORT:-30000}"
revision_args=()
if [[ -n "${REVISION:-}" ]]; then
  revision_args=(--revision "$REVISION")
fi

export SGLANG_EXTERNAL_MODEL_PACKAGE="${SGLANG_EXTERNAL_MODEL_PACKAGE:-flint_sglang}"
export FLASHINFER_USE_CUDA_NORM="${FLASHINFER_USE_CUDA_NORM:-1}"
# One Triton cache per replica: replicas compiling the same kernels into one cache directory can race.
export TRITON_CACHE_DIR="${FLINT_TRITON_CACHE_BASE:-${HOME:-/tmp}/.triton/cache}/flint-$(hostname)-${PORT}"

exec python3 -m sglang.launch_server \
  --model-path "$MODEL" "${revision_args[@]}" \
  --served-model-name flint \
  --host 0.0.0.0 --port "$PORT" \
  --tp-size 1 \
  --attention-backend fa4 \
  --moe-runner-backend flashinfer_trtllm_routed \
  --enable-fp32-lm-head \
  --context-length 262144 \
  --chunked-prefill-size 4096 \
  --swa-full-tokens-ratio 0.05 \
  --mem-fraction-static 0.8 \
  --cuda-graph-backend-prefill disabled \
  --disable-flashinfer-autotune \
  --tool-call-parser flint \
  --reasoning-parser flint \
  --grammar-backend flint \
  "$@"
