IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## Testing Rules

### Use and extend the existing test infrastructure

When adding new functionality (ops, quantization types, backends), **always use the existing test frameworks first**:

- **`test-backend-ops`** (`tests/test-backend-ops.cpp`) — the primary test framework for all ggml operations. Add new types to `all_types[]` and/or `base_types[]`. Run with `-b Vulkan`, `-b CPU`, `-o MUL_MAT`, etc. This framework handles:
  - Buffer allocation across backends
  - CPU reference computation
  - Automatic GPU-vs-CPU comparison
  - NMSE tolerance checking
  - All matrix dimension combinations
  
  **Do NOT write standalone GPU test files when test-backend-ops can test the same thing.** Standalone tests like `test-turbo-kv-gpu-roundtrip.cpp` exist only for operations that test-backend-ops cannot express (e.g., flash attention with custom KV cache layouts).

- **`test-backend-ops` is the correctness gate** — a new quantization type is not done until `test-backend-ops -b Vulkan -o MUL_MAT` passes for that type. This tests the full pipeline: quantize on CPU, upload to GPU, mul_mat on GPU, compare result against CPU.

- **Extend, don’t duplicate.** If the existing framework is missing a test case you need, add it to the framework rather than writing a parallel test.

### Shader debugging via backend-ops

When a Vulkan shader crashes or produces wrong results:
1. First verify CPU passes: `test-backend-ops -b CPU -o MUL_MAT -p turbo_4b`
2. Then test GPU: `test-backend-ops -b Vulkan -o MUL_MAT -p turbo_4b`
3. The NMSE comparison pinpoints which operations diverge
4. Use `GGML_VK_PERF_LOGGER=1` for dispatch-level profiling
