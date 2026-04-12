#!/usr/bin/env bash
# profile-gdn-vs-attn.sh — Fixture 3: GDN profile gate
#
# Determines whether the ggml_vec_dot_f16 hotspot (32.6% self-time)
# comes from GDN state matmul (f32 via f16 intermediate?) or from
# the 10 full-attention layers' KQ dot (f16 KV cache).
#
# Decision:
#   f32 self-time > 10%  →  GDN is the hotspot (proceed with Lever 3)
#   f16 self-time > 25% AND f32 < 5%  →  attention is hotspot (Lever 1 handles, skip Lever 3)
#
# Usage: bash tests/profile-gdn-vs-attn.sh
# Requires: perf, llama-server binary, 35B model

set -e
cd "$(dirname "$0")/.."

MODEL=/home/llm/models/Qwen3.5-35B-A3B-mtp-q4km.gguf
PORT=9099
PROFILE=/tmp/gdn_profile.data

echo "Starting server..."
OMP_WAIT_POLICY=ACTIVE build-tq/bin/llama-server \
    -m "$MODEL" --numa mirror -c 4096 -ngl 0 -fa off -np 1 -t 12 \
    --host 127.0.0.1 --port $PORT --no-warmup > /tmp/gdn_profile_srv.log 2>&1 &
SRV=$!
for i in $(seq 1 30); do
    if grep -q "server is listening" /tmp/gdn_profile_srv.log 2>/dev/null; then break; fi
    sleep 1
done

echo "Attaching perf..."
perf record -F 997 -g -p $SRV -o "$PROFILE" --call-graph dwarf,4096 -- sleep 30 &
PERF=$!
sleep 2

echo "Generating tokens..."
curl -s "http://127.0.0.1:$PORT/completion" \
    -d '{"prompt":"The capital of France is","n_predict":64,"temperature":0,"seed":42}' \
    > /dev/null 2>&1

wait $PERF 2>/dev/null
kill $SRV 2>/dev/null
sleep 1

echo ""
echo "=== Profile Results ==="
echo ""

F16=$(perf report -i "$PROFILE" --stdio --no-children 2>/dev/null | grep "ggml_vec_dot_f16" | head -1 | awk '{print $1}' | tr -d '%')
F32=$(perf report -i "$PROFILE" --stdio --no-children 2>/dev/null | grep "ggml_vec_dot_f32" | head -1 | awk '{print $1}' | tr -d '%')

F16=${F16:-0}
F32=${F32:-0}

echo "ggml_vec_dot_f16 self-time: ${F16}%"
echo "ggml_vec_dot_f32 self-time: ${F32}%"
echo ""

if awk "BEGIN {exit !($F32 > 10)}"; then
    echo "VERDICT: GDN is the hotspot (f32 > 10%). Proceed with Lever 3 (cache blocking)."
elif awk "BEGIN {exit !($F16 > 25 && $F32 < 5)}"; then
    echo "VERDICT: Attention is the hotspot (f16 > 25%, f32 < 5%). Lever 1 handles it. SKIP Lever 3."
else
    echo "VERDICT: Mixed signal (f16=${F16}%, f32=${F32}%). Investigate further."
fi

rm -f "$PROFILE"
echo "done"
