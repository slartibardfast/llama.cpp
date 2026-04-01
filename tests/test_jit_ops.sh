#!/bin/bash
# Systematic JIT interpreter op validation
#
# GGML_VK_JIT_OPS bitmask:
#   0=ADD 1=MUL 2=SILU 3=SIGMOID 4=SCALE 5=CPY 6=SWIGLU 7=RMS_NORM 8=L2_NORM
#
# Usage:
#   ./tests/test_jit_ops.sh             # full suite
#   ./tests/test_jit_ops.sh quick       # individual + all only
#   ./tests/test_jit_ops.sh mask 0x93   # test a single mask
#   ./tests/test_jit_ops.sh deep        # per-op CPU validation (rebuilds with CHECK_RESULTS)
#   ./tests/test_jit_ops.sh bisect 0x183  # find minimal failing subset of a mask

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
        printf "  PASS  %-50s\n" "$label"
        ((TOTAL_PASS++)); return 0
    elif [ "$result" = "__HANG__" ]; then
        printf "  HANG  %-50s\n" "$label"
        ((TOTAL_HANG++)); return 1
    else
        printf "  FAIL  %-50s\n" "$label"
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

# Collect bits set in a mask into an array
mask_bits() {
    local m=$1
    local -a bits=()
    for bit in $(seq 0 $((NUM_OPS-1))); do
        (( m & (1 << bit) )) && bits+=($bit)
    done
    echo "${bits[@]}"
}

# Test all subsets of size k from a mask. Reports failures.
# Args: mask, subset_size
test_subsets() {
    local mask=$1 size=$2
    local bits=($(mask_bits $mask))
    local n=${#bits[@]}
    local found_fail=0

    if (( size == 1 )); then
        for ((i=0; i<n; i++)); do
            local sub=$((1 << bits[i]))
            local result=$(run 1 $sub)
            if [ "$result" != "$BASELINE" ]; then
                printf "    -> %d-op FAIL: %-30s (0x%x)\n" "$size" "$(mask_name $sub)" "$sub"
                ((found_fail++))
            fi
        done
    elif (( size == 2 )); then
        for ((i=0; i<n-1; i++)); do
            for ((j=i+1; j<n; j++)); do
                local sub=$(( (1 << bits[i]) | (1 << bits[j]) ))
                local result=$(run 1 $sub)
                if [ "$result" != "$BASELINE" ]; then
                    printf "    -> %d-op FAIL: %-30s (0x%x)\n" "$size" "$(mask_name $sub)" "$sub"
                    ((found_fail++))
                fi
            done
        done
    elif (( size == 3 )); then
        for ((i=0; i<n-2; i++)); do
            for ((j=i+1; j<n-1; j++)); do
                for ((k=j+1; k<n; k++)); do
                    local sub=$(( (1 << bits[i]) | (1 << bits[j]) | (1 << bits[k]) ))
                    local result=$(run 1 $sub)
                    if [ "$result" != "$BASELINE" ]; then
                        printf "    -> %d-op FAIL: %-30s (0x%x)\n" "$size" "$(mask_name $sub)" "$sub"
                        ((found_fail++))
                    fi
                done
            done
        done
    elif (( size == 4 )); then
        for ((i=0; i<n-3; i++)); do
            for ((j=i+1; j<n-2; j++)); do
                for ((k=j+1; k<n-1; k++)); do
                    for ((l=k+1; l<n; l++)); do
                        local sub=$(( (1 << bits[i]) | (1 << bits[j]) | (1 << bits[k]) | (1 << bits[l]) ))
                        local result=$(run 1 $sub)
                        if [ "$result" != "$BASELINE" ]; then
                            printf "    -> %d-op FAIL: %-30s (0x%x)\n" "$size" "$(mask_name $sub)" "$sub"
                            ((found_fail++))
                        fi
                    done
                done
            done
        done
    fi
    return $found_fail
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

# --- Mode: single mask test ---
if [ "${1:-}" = "mask" ]; then
    result=$(run 1 "$2")
    check "mask=$2 ($(mask_name $(($2))))" "$result"
    exit 0
fi

# --- Mode: bisect a failing mask to minimal subset ---
if [ "${1:-}" = "bisect" ]; then
    MASK=$(($2))
    echo "--- Bisecting mask 0x$(printf '%x' $MASK) ($(mask_name $MASK)) ---"
    result=$(run 1 $MASK)
    if [ "$result" = "$BASELINE" ]; then
        echo "  mask 0x$(printf '%x' $MASK) PASSES — nothing to bisect"
        exit 0
    fi
    echo "  Confirmed: mask 0x$(printf '%x' $MASK) FAILS"
    echo ""

    bits=($(mask_bits $MASK))
    n=${#bits[@]}

    for size in 1 2 3 4; do
        (( size > n )) && break
        local_count=0
        # Count combos at this size
        case $size in
            1) combos=$n ;;
            2) combos=$((n*(n-1)/2)) ;;
            3) combos=$((n*(n-1)*(n-2)/6)) ;;
            4) combos=$((n*(n-1)*(n-2)*(n-3)/24)) ;;
        esac
        echo "  Testing ${size}-op subsets ($combos combos)..."
        test_subsets $MASK $size
        failures=$?
        if (( failures > 0 )); then
            echo "  Found $failures failing ${size}-op subset(s) — minimal failure size is $size"
            exit 0
        fi
        echo "    All ${size}-op subsets pass"
    done

    # If we get here, check N-1 subsets (drop one op at a time)
    echo "  Testing (N-1)-op subsets (drop one at a time)..."
    for bit in ${bits[@]}; do
        sub=$((MASK ^ (1 << bit)))
        result=$(run 1 $sub)
        if [ "$result" = "$BASELINE" ]; then
            printf "    drop %-10s → PASS  (0x%x) — %s is REQUIRED for failure\n" "${OP_NAMES[$bit]}" "$sub" "${OP_NAMES[$bit]}"
        else
            printf "    drop %-10s → FAIL  (0x%x) — %s is NOT required\n" "${OP_NAMES[$bit]}" "$sub" "${OP_NAMES[$bit]}"
        fi
    done
    exit 0
fi

# --- Mode: deep per-op CPU validation ---
if [ "${1:-}" = "deep" ]; then
    echo "--- Deep: Per-op CPU vs GPU validation ---"
    echo "  Building with GGML_VULKAN_CHECK_RESULTS=ON..."
    cd build
    cmake -DGGML_VULKAN_CHECK_RESULTS=ON .. >/dev/null 2>&1
    rm -f ggml/src/ggml-vulkan/CMakeFiles/ggml-vulkan.dir/ggml-vulkan.cpp.o
    cmake --build . --target llama-completion -j$(nproc) 2>&1 | tail -1

    deep_pass=0; deep_fail=0
    for mode_label in "JIT_OFF" "JIT_ON"; do
        [ "$mode_label" = "JIT_OFF" ] && jit=0 || jit=1
        echo ""
        echo "  [$mode_label] Validating all ops against CPU reference..."
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

# --- Full test suite ---

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

# Phase 4: Cumulative — add ops one at a time
echo "--- Phase 4: Cumulative ---"
cumul=0
CUMUL_FAIL_MASK=0
for bit in 0 1 4 7 8 2 3 6 5; do  # ADD MUL SCALE RMS L2 SILU SIGMOID SWIGLU CPY
    cumul=$((cumul | (1 << bit)))
    name=${OP_NAMES[$bit]}
    result=$(run 1 "$cumul")
    if ! check "+$name (0x$(printf '%x' $cumul))" "$result"; then
        CUMUL_FAIL_MASK=$cumul
        break
    fi
done
echo ""

# Phase 5: Auto-bisect if cumulative failed
if (( CUMUL_FAIL_MASK > 0 )); then
    echo "--- Phase 5: Auto-bisect of 0x$(printf '%x' $CUMUL_FAIL_MASK) ---"
    bits=($(mask_bits $CUMUL_FAIL_MASK))
    n=${#bits[@]}

    for size in 1 2 3 4; do
        (( size > n )) && break
        case $size in
            1) combos=$n ;;
            2) combos=$((n*(n-1)/2)) ;;
            3) combos=$((n*(n-1)*(n-2)/6)) ;;
            4) combos=$((n*(n-1)*(n-2)*(n-3)/24)) ;;
        esac
        echo "  Testing ${size}-op subsets ($combos tests)..."
        test_subsets $CUMUL_FAIL_MASK $size
        failures=$?
        if (( failures > 0 )); then
            echo "  → Minimal failing subset size: $size"
            echo ""
            # Also show which ops are required by dropping one at a time
            echo "  Required ops (drop-one-at-a-time from full failing mask):"
            for bit in ${bits[@]}; do
                sub=$((CUMUL_FAIL_MASK ^ (1 << bit)))
                result=$(run 1 $sub)
                if [ "$result" = "$BASELINE" ]; then
                    printf "    drop %-10s → PASS — %s REQUIRED\n" "${OP_NAMES[$bit]}" "${OP_NAMES[$bit]}"
                else
                    printf "    drop %-10s → FAIL — %s not required\n" "${OP_NAMES[$bit]}" "${OP_NAMES[$bit]}"
                fi
            done
            break
        fi
        echo "    All ${size}-op subsets pass"
    done
    echo ""
fi

# Phase 6: All ops
echo "--- Phase 6: All ops (0x1ff) ---"
check "all ops" "$(run 1 0x1ff)"
echo ""

echo "=== Summary: $TOTAL_PASS pass, $TOTAL_FAIL fail, $TOTAL_HANG hang ==="
