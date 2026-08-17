#!/usr/bin/env bash
# Clean-and-test wrapper for the Q4_0_AR16 surface (private fork, branch autoround).
#
# Operationalises the rule: "never test outdated objects." The build is a
# MANDATORY precondition of every test run. cmake --build is correct-by-
# construction (ninja rebuilds exactly the objects whose sources or headers
# changed), and `cmake -B` is re-run each time so a changed CMake option or
# new source file is always picked up. There is no path that runs a test
# against a binary older than its sources.
#
# Usage:  bash tests/run-ar16-tests.sh
#   build dir is ./build; the configure options are pinned below and are the
#   only reproduction contract (see the AR16 rebase / rule-taker notes).
#
# Env overrides (all optional):
#   AR16_BUILD_DIR   - build directory (default ./build)
#   AR16_JOBS        - parallelism (default nproc)
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR="${AR16_BUILD_DIR:-build}"
JOBS="${AR16_JOBS:-$(nproc)}"

# ---- 1. Configure (idempotent: no-op if the cache already matches). ----
# These are the exact options of the reproduced CUDA build (sm_75, Release).
cmake -S . -B "$BUILD_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_NCCL=ON

# ---- 2. Build (correct-by-construction: ninja only rebuilds stale objects). ----
cmake --build "$BUILD_DIR" -j "$JOBS" --target \
    test-q4-0-ar16 \
    test-backend-ops \
    llama-server

# ---- 3. Test. ----
echo "== test-q4-0-ar16 (dequant/tie/SIMD/MUL_MAT/MUL_MAT_ID/split) =="
"$BUILD_DIR/bin/test-q4-0-ar16"

echo "== test-backend-ops, full CPU sweep (incl. q4_0_ar16 ops) =="
"$BUILD_DIR/bin/test-backend-ops" -b CPU

echo "== all AR16 clean-build checks passed =="
