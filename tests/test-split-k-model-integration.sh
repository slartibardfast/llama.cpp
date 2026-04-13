#!/usr/bin/env bash
# test-split-k-model-integration.sh — model-level split K validation
#
# Tests the actual model at various context lengths to reproduce
# the c=2048 get_rows OOB crash. Uses llama-completion (not perplexity)
# with a long prompt to fill the context.
#
# Usage: bash tests/test-split-k-model-integration.sh [model_path]

set -e
cd "$(dirname "$0")/.."

BIN="build-tq/bin"
MODEL="${1:-/home/llm/models/Qwen3.5-27B-mtp-toolcall.gguf}"

if [ ! -f "$MODEL" ]; then
    echo "Model not found: $MODEL"
    exit 1
fi

echo "=== Split K Model Integration Tests ==="
echo "Model: $(basename $MODEL)"
echo ""

PASS=0; FAIL=0; CRASH=0

run_test() {
    local desc="$1"
    local ck="$2"
    local cv="$3"
    local ctx="$4"
    local prompt_words="$5"
    local n_predict="$6"

    # Generate a prompt of the right length
    local PROMPT=$(python3 -c "print(' '.join(['word'] * $prompt_words))")

    printf "  %-60s " "$desc"

    local OUTPUT
    OUTPUT=$($BIN/llama-completion \
        -m "$MODEL" \
        -p "$PROMPT" -n "$n_predict" -c "$ctx" \
        --simple-io -no-cnv --temp 0 \
        --cache-type-k "$ck" --cache-type-v "$cv" \
        --no-warmup -ngl 0 -t 12 --numa mirror --seed 42 \
        2>&1) || true

    if echo "$OUTPUT" | grep -q "GGML_ASSERT"; then
        echo "CRASH: $(echo "$OUTPUT" | grep GGML_ASSERT | head -1)"
        CRASH=$((CRASH + 1))
    elif echo "$OUTPUT" | grep -q "error"; then
        echo "ERROR"
        CRASH=$((CRASH + 1))
    else
        echo "OK"
        PASS=$((PASS + 1))
    fi
}

echo "--- 1. Baseline f16 (should all pass) ---"
run_test "f16, c=512, short prompt"          f16 f16  512  10  5
run_test "f16, c=512, fill context"          f16 f16  512 400  5
run_test "f16, c=1024, fill context"         f16 f16 1024 800  5
run_test "f16, c=2048, fill context"         f16 f16 2048 1600 5
run_test "f16, c=4096, fill context"         f16 f16 4096 3200 5

echo ""
echo "--- 2. Split K q8_0:q4_0 / f16 ---"
run_test "split K, c=512, short"             q8_0:q4_0 f16  512  10  5
run_test "split K, c=512, fill"              q8_0:q4_0 f16  512 400  5
run_test "split K, c=1024, fill"             q8_0:q4_0 f16 1024 800  5
run_test "split K, c=2048, short"            q8_0:q4_0 f16 2048  10  5
run_test "split K, c=2048, half fill"        q8_0:q4_0 f16 2048 800  5
run_test "split K, c=2048, fill"             q8_0:q4_0 f16 2048 1600 5
run_test "split K, c=4096, fill"             q8_0:q4_0 f16 4096 3200 5

echo ""
echo "--- 3. Split K + turbo V ---"
run_test "split+turbo, c=512, fill"          q8_0:q4_0 turbo_kv_4b  512 400  5
run_test "split+turbo, c=1024, fill"         q8_0:q4_0 turbo_kv_4b 1024 800  5
run_test "split+turbo, c=2048, fill"         q8_0:q4_0 turbo_kv_4b 2048 1600 5

echo ""
echo "=== Results: $PASS pass, $CRASH crash/error ==="
