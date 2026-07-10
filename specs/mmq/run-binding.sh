#!/usr/bin/env bash
# Build + run the q4_0_ar16 binding fixture against this tree's ggml.
# CPU-only static build; ccache stays OFF (ccache 4.13.6 segfaults these builds).
# Compiler override: CC=gcc-15 CXX=g++-15 ./specs/mmq/run-binding.sh (defaults: cc/c++).
set -euo pipefail
cd "$(dirname "$0")/../.."

CC="${CC:-cc}"
CXX="${CXX:-c++}"

cmake -B build -G Ninja \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
    -DGGML_CCACHE=OFF -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF
cmake --build build --target ggml-cpu

"$CC" -O2 -Wall -Iggml/include -Iggml/src specs/mmq/q4_0_ar16_binding.c \
    build/ggml/src/libggml-cpu.a build/ggml/src/libggml-base.a \
    -lstdc++ -lm -fopenmp -o build/q4_0_ar16_binding

./build/q4_0_ar16_binding
