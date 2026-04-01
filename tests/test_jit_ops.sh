#!/bin/bash
# Systematic JIT interpreter op validation
#
# GGML_VK_JIT_OPS bitmask:
#   0=ADD 1=MUL 2=SILU 3=SIGMOID 4=SCALE 5=CPY 6=SWIGLU 7=RMS_NORM 8=L2_NORM
#
# Usage:
#   ./tests/test_jit_ops.sh           # full suite (individual + pairs + cumulative + all)
#   ./tests/test_jit_ops.sh quick     # individual + all only
#   ./tests/test_jit_ops.sh mask 0x93 # test a single mask
#   ./tests/test_jit_ops.sh deep      # build with CHECK_RESULTS, per-op CPU validation

set -uo pipefail
cd "$(dirname "$0")/.."

BIN=build/bin/llama-completion
MODEL=${JIT_TEST_MODEL:-/home/llm/models/Qwen3.5-0.8B-Q4_K_M.gguf}
PROMPT=${JIT_TEST_PROMPT:-The meaning of life is}
NTOKENS=${JIT_TEST_N:-20}
ARGS="-p '$PROMPT' -n $NTOKENS -s 42 --simple-io -no-cnv --no-warmup -ngl 100 -c 256"
TIMEOUT=${JIT_TEST_TIMEOUT:-30}

NUM_OPS=9
declare -A OP_NAMES=(
    [0]="ADD" [1]="MUL" [2]="SILU" [3]="SIGMOID"
    [4]="SCALE" [5]="CPY" [6]="SWIGLU" [7]="RMS_NORM" [8]="L2_NORM"
)

TOTAL_PASS=0; TOTAL_FAIL=0; TOTAL_HANG=0

run() {
    local jit=$1 ops=${2:-}
    local env="GGML_VK_JIT=$jit"
    [ -n "$ops" ] && env="$env GGML_VK_JIT_OPS=$ops"
    local out
    out=$(eval "timeout $TIMEOUT env $env $BIN -m $MODEL $ARGS 2>/dev/null") || true
    [ -z "$out" ] && out="__HANG__"
    echo "$out"
}

check() {
    local label=$1 result=$2
    if [ "$result" = "$BASELINE" ]; then
        printf "  PASS  %-45s\n" "$label"
        ((TOTAL_PASS++)); return 0
    elif [ "$result" = "__HANG__" ]; then
        printf "  HANG  %-45s\n" "$label"
        ((TOTAL_HANG++)); return 1
    else
        printf "  FAIL  %-45s\n" "$label"
        printf "    got: %.80s\n" "$result"
        ((TOTAL_FAIL++)); return 1
    fi
}

mask_name() {
    local m=$1 names=""
    for bit in $(seq 0 $((NUM_OPS-1))); do
        (( m & (1 << bit) )) && { [ -n "$names" ] && names="$names+"; names="$names${OP_NAMES[$bit]}"; }
    done
    echo "${names:-NONE}"
}

echo "=== JIT Op Validation ==="
echo "Model: $(basename "$MODEL")"
echo "Prompt: '$PROMPT' n=$NTOKENS seed=42 timeout=${TIMEOUT}s"
echo ""

# Phase 1: Baseline
echo "--- Phase 1: Baseline ---"
BASELINE=$(run 0)
[ "$BASELINE" = "__HANG__" ] && { echo "  ERROR: baseline hangs"; exit 1; }
echo "  Output: ${BASELINE:0:80}..."
echo ""

# Single mask test mode
if [ "${1:-}" = "mask" ]; then
    result=$(run 1 "$2")
    check "mask=$2 ($(mask_name $(($2))))" "$result"
    exit 0
fi

# Deep per-op CPU validation mode
if [ "${1:-}" = "deep" ]; then
    echo "--- Deep: Per-op CPU vs GPU validation ---"
    echo "  Building with GGML_VULKAN_CHECK_RESULTS=ON..."
    cd build
    cmake -DGGML_VULKAN_CHECK_RESULTS=ON .. >/dev/null 2>&1
    cmake --build . --target llama-completion -j$(nproc) 2>&1 | tail -1

    deep_pass=0; deep_fail=0
    for mode_label in "JIT_OFF" "JIT_ON"; do
        [ "$mode_label" = "JIT_OFF" ] && jit=0 || jit=1
        echo ""
        echo "  [$mode_label] Validating all ops against CPU reference..."
        # CHECK_RESULTS aborts on first error (avg_err > 0.01)
        # Capture stderr for validation output
        local_out=$(timeout 120 env GGML_VK_JIT=$jit \
            ./bin/llama-completion -m "$MODEL" \
            -p "$PROMPT" -n 1 -s 42 --simple-io -no-cnv --no-warmup -ngl 100 -c 256 \
            2>&1 >/dev/null) || true
        aborts=$(echo "$local_out" | grep -c "GGML_ABORT\|avg_err" || true)
        if [ "$aborts" -gt 0 ]; then
            echo "  FAIL $mode_label: validation errors detected"
            echo "$local_out" | grep -E "avg_err|ABORT|first_error" | tail -10
            ((deep_fail++))
        else
            echo "  PASS $mode_label: all ops within tolerance (avg_err < 0.01)"
            ((deep_pass++))
        fi
    done

    echo ""
    echo "  Restoring normal build (force recompile)..."
    cmake -DGGML_VULKAN_CHECK_RESULTS=OFF .. >/dev/null 2>&1
    rm -f ggml/src/ggml-vulkan/CMakeFiles/ggml-vulkan.dir/ggml-vulkan.cpp.o
    cmake --build . --target llama-completion -j$(nproc) 2>&1 | tail -1
    cd ..
    echo ""
    echo "=== Deep: $deep_pass pass, $deep_fail fail ==="
    exit $deep_fail
fi

# Phase 2: Individual ops (9 tests)
echo "--- Phase 2: Individual (9 tests) ---"
for bit in $(seq 0 $((NUM_OPS-1))); do
    mask=$((1 << bit))
    check "${OP_NAMES[$bit]} (0x$(printf '%x' $mask))" "$(run 1 $mask)"
done
echo ""

[ "${1:-}" = "quick" ] && {
    echo "--- Quick: All ops ---"
    check "all (0x1ff)" "$(run 1 0x1ff)"
    echo ""
    echo "=== $TOTAL_PASS pass, $TOTAL_FAIL fail, $TOTAL_HANG hang ==="
    exit 0
}

# Phase 3: All pairs (36 tests)
echo "--- Phase 3: Pairs (36 tests) ---"
for a in $(seq 0 $((NUM_OPS-2))); do
    for b in $(seq $((a+1)) $((NUM_OPS-1))); do
        mask=$(( (1 << a) | (1 << b) ))
        check "${OP_NAMES[$a]}+${OP_NAMES[$b]} (0x$(printf '%x' $mask))" "$(run 1 $mask)"
    done
done
echo ""

# Phase 4: Cumulative — add ops one at a time, auto-bisect on failure
echo "--- Phase 4: Cumulative ---"
cumul=0
for bit in 0 1 4 7 8 2 3 6 5; do  # ADD MUL SCALE RMS L2 SILU SIGMOID SWIGLU CPY
    cumul=$((cumul | (1 << bit)))
    name=${OP_NAMES[$bit]}
    result=$(run 1 "$cumul")
    if ! check "+$name (0x$(printf '%x' $cumul))" "$result"; then
        echo "    Bisecting: which prior op conflicts with $name?"
        for prev_bit in 0 1 4 7 8 2 3 6 5; do
            [ $prev_bit -eq $bit ] && break
            pair=$(( (1 << bit) | (1 << prev_bit) ))
            pair_result=$(run 1 $pair)
            [ "$pair_result" != "$BASELINE" ] && echo "    -> ${OP_NAMES[$prev_bit]}+$name (0x$(printf '%x' $pair))"
        done
        break
    fi
done
echo ""

# Phase 5: All ops
echo "--- Phase 5: All ops (0x1ff) ---"
check "all ops" "$(run 1 0x1ff)"
echo ""

echo "=== Summary: $TOTAL_PASS pass, $TOTAL_FAIL fail, $TOTAL_HANG hang ==="
