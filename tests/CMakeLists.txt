llama_add_compile_flags()

function(llama_build source)
    set(TEST_SOURCES ${source} ${ARGN})

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE common)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_test_cmd target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS} get-model.cpp)

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE common)

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gemma-4           ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
endif ()

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-sampling.cpp)
    llama_build_and_test(test-reasoning-budget.cpp)
    llama_build_and_test(test-grammar-parser.cpp)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp)
    llama_build_and_test(test-chat.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    # TODO: disabled on loongarch64 because the ggml-ci node lacks Python 3.8
    if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
        llama_build_and_test(test-json-schema-to-grammar.cpp   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        target_include_directories(test-json-schema-to-grammar PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    endif()

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW

    llama_build_and_test(test-llama-archs.cpp)
endif()

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp)
llama_build_and_test(test-json-partial.cpp)
llama_build_and_test(test-log.cpp)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-json-serialization.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
)
llama_build_and_test(test-regex-partial.cpp)

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    set(MODEL_NAME "tinyllamas/stories15M-q4_0.gguf")
    set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
    set(MODEL_NAME "tinyllamas/stories15M-be.Q4_0.gguf")
    set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")

add_test(NAME test-download-model COMMAND ${CMAKE_COMMAND}
    -DDEST=${MODEL_DEST}
    -DNAME=${MODEL_NAME}
    -DHASH=${MODEL_HASH}
    -P ${CMAKE_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

llama_build_and_test(test-thread-safety.cpp ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-arg-parser.cpp)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp)
endif()
llama_build_and_test(test-gguf.cpp)
llama_build_and_test(test-backend-ops.cpp)

llama_build_and_test(test-model-load-cancel.cpp LABEL "model")
llama_build_and_test(test-autorelease.cpp       LABEL "model")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model")

# Test for state restore with fragmented KV cache
# Requires a model, uses same args pattern as test-thread-safety
llama_build_and_test(test-state-restore-fragmented.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-state-restore-fragmented PROPERTIES FIXTURES_REQUIRED test-download-model)

# Recurrent snapshot/restore for hybrid DeltaNet models (needs a hybrid model).
llama_build_and_test(test-mtp-snapshot.cpp LABEL "manual-hybrid")

# TurboQuant KV cache correctness fixtures (no model needed — pure math / synthetic data).
add_executable(test-turbo-kv-rht test-turbo-kv-rht.cpp)
target_link_libraries(test-turbo-kv-rht PRIVATE ggml-base)

add_executable(test-turbo-kv-vulkan test-turbo-kv-vulkan.cpp)
target_link_libraries(test-turbo-kv-vulkan PRIVATE ggml-base ggml)

add_executable(test-turbo-kv-gpu-roundtrip test-turbo-kv-gpu-roundtrip.cpp)
target_link_libraries(test-turbo-kv-gpu-roundtrip PRIVATE ggml-base ggml)

add_executable(test-turbo-kv-gpu-quantize test-turbo-kv-gpu-quantize.cpp)
target_link_libraries(test-turbo-kv-gpu-quantize PRIVATE ggml-base ggml)

add_executable(test-turbo-kv-set-rows test-turbo-kv-set-rows.cpp)
target_link_libraries(test-turbo-kv-set-rows PRIVATE ggml-base ggml)

add_executable(test-turbo-kv-4b-attn test-turbo-kv-4b-attn.cpp)
target_link_libraries(test-turbo-kv-4b-attn PRIVATE ggml-base)

add_executable(test-turbo-4b-roundtrip test-turbo-4b-roundtrip.cpp)
target_link_libraries(test-turbo-4b-roundtrip PRIVATE ggml-base)

add_executable(test-turbo-4b-codebook test-turbo-4b-codebook.cpp)
target_link_libraries(test-turbo-4b-codebook PRIVATE ggml-base ggml)

add_executable(test-turbo-4b-e8p test-turbo-4b-e8p.cpp)
target_link_libraries(test-turbo-4b-e8p PRIVATE ggml-base)

# TurboQuant PBT: property-based tests from turbo-kv-4b.allium and
# mul_mat_cpu.allium. The SIMDEquivalence property from mul_mat_cpu
# drives a minimal CPU-backend mul_mat graph, so we link ggml-cpu in
# addition to ggml-base.
add_executable(test-turbo-kv-pbt test-turbo-kv-pbt.cpp)
target_link_libraries(test-turbo-kv-pbt PRIVATE ggml-base ggml-cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rapidcheck/build/librapidcheck.a)
target_include_directories(test-turbo-kv-pbt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rapidcheck/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
# -march=native so __AVX2__ is defined and the AVX2 differential
# property in test-turbo-kv-pbt.cpp pulls in turbo_kv_4b_avx2.h.
target_compile_options(test-turbo-kv-pbt PRIVATE -march=native)

# TurboQuant backend PBT: propagated from turbo_kv_4b_backend.allium.
# Tests any registered non-CPU backend's quantize / dequantize /
# get_rows / set_rows for TURBO_KV_4B against the CPU scalar reference
# within the fp32 tolerance floor. Iterates over ggml_backend_dev_get()
# at runtime so adding CUDA / ROCm / Metal to the build surfaces them
# through enumeration without a file edit here. Links ggml (the full
# backend library) so Vulkan / CUDA / etc. shader dispatch is
# available.
add_executable(test-turbo-kv-backend-pbt test-turbo-kv-backend-pbt.cpp)
target_link_libraries(test-turbo-kv-backend-pbt PRIVATE ggml-base ggml
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rapidcheck/build/librapidcheck.a)
target_include_directories(test-turbo-kv-backend-pbt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rapidcheck/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)

# TurboQuant residual-window PBT: propagated from
# turbo_kv_residual_window.allium. Establishes the full 22-obligation
# coverage list against the spec. Residual window is not yet
# implemented in the port (PHASE26 Tier 1.1 — primary driver of the
# 9B IQ3_XXS PPL regression), so implementation-dependent obligations
# SKIP today. Builds and runs green; converts SKIPs to rc::check
# calls as the feature lands.
add_executable(test-turbo-kv-residual-window-pbt test-turbo-kv-residual-window-pbt.cpp)
target_link_libraries(test-turbo-kv-residual-window-pbt PRIVATE ggml-base)
target_include_directories(test-turbo-kv-residual-window-pbt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
# rapidcheck linkage will be added when the first rc::check lands
# (today the file is skip-only — see header comment).

# TurboQuant asymmetric-K/V PBT: propagated from
# turbo_kv_asymmetric.allium. PHASE26 Tier 1.2. Skip-stub today.
add_executable(test-turbo-kv-asymmetric-pbt test-turbo-kv-asymmetric-pbt.cpp)
target_link_libraries(test-turbo-kv-asymmetric-pbt PRIVATE ggml-base)

# TurboQuant layer-adaptive PBT: propagated from
# turbo_kv_layer_adaptive.allium. PHASE26 Tier 2.1. Skip-stub today.
add_executable(test-turbo-kv-layer-adaptive-pbt test-turbo-kv-layer-adaptive-pbt.cpp)
target_link_libraries(test-turbo-kv-layer-adaptive-pbt PRIVATE ggml-base)

# TurboQuant outliers PBT: propagated from turbo_kv_outliers.allium.
# PHASE26 Tier 2.2. Skip-stub today.
add_executable(test-turbo-kv-outliers-pbt test-turbo-kv-outliers-pbt.cpp)
target_link_libraries(test-turbo-kv-outliers-pbt PRIVATE ggml-base)

# TurboQuant attention PBT: propagated from turbo_kv_4b_attention.allium.
# Compares GGML_OP_FLASH_ATTN_EXT output between the CPU scalar
# reference and each registered non-CPU backend at the multi-block
# head_dim=256 configuration used by Qwen3.5 9B / 35B-A3B. Diagnoses
# the 9B PPL regression — if this passes within spec tolerance, the
# regression is algorithmic (codebook granularity at multi-block
# heads); if it fails, there's a latent GPU FA bug analogous to the
# CPY / GET_ROWS multi-block bugs fixed earlier.
add_executable(test-turbo-kv-attention-pbt test-turbo-kv-attention-pbt.cpp)
target_link_libraries(test-turbo-kv-attention-pbt PRIVATE ggml-base ggml
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rapidcheck/build/librapidcheck.a)
target_include_directories(test-turbo-kv-attention-pbt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rapidcheck/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)

# Micro-benchmark for PHASE25 step 3 profiling — isolates
# turbo_kv_4b_avx2_single_block_dot in a tight loop so `perf record`
# attribution lands on the kernel's hot instructions, not harness noise.
# Needs -march=native explicitly because it pulls in the AVX2 kernel
# header directly (bypasses the ggml-cpu trait dispatch that already
# guards on __AVX2__).
add_executable(bench-turbo-kv-4b-avx2 bench-turbo-kv-4b-avx2.cpp)
target_link_libraries(bench-turbo-kv-4b-avx2 PRIVATE ggml-base)
target_include_directories(bench-turbo-kv-4b-avx2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
target_compile_options(bench-turbo-kv-4b-avx2 PRIVATE -march=native)

# Sibling micro-bench for the quantize path (PHASE25 step 3 follow-up).
# Hammers quantize_row_turbo_kv_4b_ref which is the exported API that
# runs per-token-per-layer during prefill AND decode KV writes.
# No arch-specific header here — quantize_row_turbo_kv_4b_ref is
# ggml-base's exported symbol; the SSE4.1 walsh_hadamard_sse inside
# it is already guarded on __SSE4_1__.
add_executable(bench-turbo-kv-quantize bench-turbo-kv-quantize.cpp)
target_link_libraries(bench-turbo-kv-quantize PRIVATE ggml-base ggml-cpu)

# AVX2 argmin head-to-head: same (rotated, inv_std) pool, scalar vs
# AVX2 Step-5 only, reports ns/call and speedup. PHASE25 step 3
# follow-up to measure the vectorized-argmin optimization target.
add_executable(bench-turbo-kv-argmin bench-turbo-kv-argmin.cpp)
target_link_libraries(bench-turbo-kv-argmin PRIVATE ggml-base)
target_include_directories(bench-turbo-kv-argmin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
target_compile_options(bench-turbo-kv-argmin PRIVATE -march=native)

add_executable(test-split-k-attention test-split-k-attention.cpp)

add_executable(test-split-k-graph test-split-k-graph.cpp)
target_link_libraries(test-split-k-graph PRIVATE ggml-base ggml-cpu)

add_executable(test-split-k-cache-ops test-split-k-cache-ops.cpp)
target_link_libraries(test-split-k-cache-ops PRIVATE ggml-base ggml-cpu)

add_executable(test-mtp-greedy-tokens test-mtp-greedy-tokens.cpp)
target_link_libraries(test-mtp-greedy-tokens PRIVATE ggml-base ggml-cpu)

if (NOT GGML_BACKEND_DL)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp)
    llama_build_and_test(test-quantize-fns.cpp)
    llama_build_and_test(test-quantize-perf.cpp)
    llama_build_and_test(test-rope.cpp)
endif()

# libmtmd
set(LLAMA_TEST_NAME test-mtmd-c-api)
llama_build_and_test(test-mtmd-c-api.c)
target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)
unset(LLAMA_TEST_NAME)

# GGUF model data fetcher library for tests that need real model metadata
# Only compile when cpp-httplib has SSL support (CPPHTTPLIB_OPENSSL_SUPPORT)
if (TARGET cpp-httplib)
    get_target_property(_cpp_httplib_defs cpp-httplib INTERFACE_COMPILE_DEFINITIONS)
    if (_cpp_httplib_defs MATCHES "CPPHTTPLIB_OPENSSL_SUPPORT")
        add_library(gguf-model-data STATIC gguf-model-data.cpp)
        target_link_libraries(gguf-model-data PRIVATE common cpp-httplib)
        target_include_directories(gguf-model-data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

        add_executable(test-gguf-model-data test-gguf-model-data.cpp)
        target_link_libraries(test-gguf-model-data PRIVATE gguf-model-data common)
        llama_test(test-gguf-model-data LABEL "model")

        # test-quant-type-selection requires gguf-model-data for remote model metadata
        llama_build_and_test(test-quant-type-selection.cpp LABEL "model")
        target_link_libraries(test-quant-type-selection PRIVATE gguf-model-data)
        target_compile_definitions(test-quant-type-selection PRIVATE
            SNAPSHOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/snapshots")
    endif()
endif()

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

llama_build_and_test(test-alloc.cpp)
target_include_directories(test-alloc PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)

llama_build(export-graph-ops.cpp)
target_include_directories(export-graph-ops PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
if (TARGET gguf-model-data)
    target_link_libraries(export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()
