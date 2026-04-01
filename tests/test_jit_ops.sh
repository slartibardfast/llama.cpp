#!/bin/bash
# Systematic JIT interpreter op validation
# Tests each op type independently against baseline, then combinations
#
# GGML_VK_JIT_OPS bitmask:
#   0=ADD 1=MUL 2=SILU 3=SIGMOID 4=SCALE 5=CPY 6=SWIGLU 7=RMS_NORM 8=L2_NORM

set -uo pipefail
cd "$(dirname "$0")/.."

BIN=build/bin/llama-completion
MODEL=/home/llm/models/Qwen3.5-0.8B-Q4_K_M.gguf
ARGS="-p 'The meaning of life is' -n 20 -s 42 --simple-io -no-cnv --no-warmup -ngl 100 -c 256"
TIMEOUT=60

declare -A OP_NAMES=(
    [0]="ADD" [1]="MUL" [2]="SILU" [3]="SIGMOID"
    [4]="SCALE" [5]="CPY" [6]="SWIGLU" [7]="RMS_NORM" [8]="L2_NORM"
)

run() {
    local label=$1 jit=$2 ops=${3:-}
    local env="GGML_VK_JIT=$jit"
    [ -n "$ops" ] && env="$env GGML_VK_JIT_OPS=$ops"
    local out
    out=$(eval "timeout $TIMEOUT env $env $BIN -m $MODEL $ARGS 2>/dev/null") || true
    [ -z "$out" ] && out="TIMEOUT_OR_ERROR"
    echo "$out"
}

echo "=== JIT Op Validation ==="
echo "Model: $MODEL"
echo ""

# Phase 1: Baseline
echo "--- Phase 1: Baseline ---"
BASELINE=$(run "baseline" 0)
echo "Baseline: ${BASELINE:0:80}..."
echo ""

# Phase 2: Each op individually (mask = 1 << bit)
echo "--- Phase 2: Individual ops ---"
PASS=0; FAIL=0; SKIP=0
for bit in 0 1 2 3 4 5 6 7 8; do
    mask=$((1 << bit))
    name=${OP_NAMES[$bit]}
    result=$(run "$name" 1 "$mask")
    if [ "$result" = "$BASELINE" ]; then
        echo "  PASS  $name (mask=$mask)"
        ((PASS++))
    elif [ "$result" = "TIMEOUT_OR_ERROR" ]; then
        echo "  HANG  $name (mask=$mask)"
        ((FAIL++))
    else
        echo "  FAIL  $name (mask=$mask)"
        echo "    got: ${result:0:80}"
        ((FAIL++))
    fi
done
echo "Individual: $PASS pass, $FAIL fail"
echo ""

# Phase 3: Cumulative — add ops one at a time
echo "--- Phase 3: Cumulative ---"
cumul=0
for bit in 0 1 4 7 8 2 3 6 5; do  # order: ADD MUL SCALE RMS SILU SIGMOID SWIGLU CPY
    cumul=$((cumul | (1 << bit)))
    name=${OP_NAMES[$bit]}
    result=$(run "+$name" 1 "$cumul")
    if [ "$result" = "$BASELINE" ]; then
        echo "  PASS  +$name (mask=0x$(printf '%x' $cumul))"
    elif [ "$result" = "TIMEOUT_OR_ERROR" ]; then
        echo "  HANG  +$name (mask=0x$(printf '%x' $cumul))"
        break
    else
        echo "  FAIL  +$name (mask=0x$(printf '%x' $cumul))"
        echo "    got: ${result:0:80}"
        break
    fi
done
echo ""

# Phase 4: All ops
echo "--- Phase 4: All ops (mask=0x1ff) ---"
result=$(run "all" 1 "0x1ff")
if [ "$result" = "$BASELINE" ]; then
    echo "  PASS  all ops"
else
    echo "  FAIL  all ops"
    echo "    got: ${result:0:80}"
fi
