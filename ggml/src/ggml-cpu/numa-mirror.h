#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_compute_params;

// NUMA-mirror CPU buffer type. Allocates two physical copies of every
// buffer it owns, one bound to each NUMA node, and exposes a per-buffer
// alt offset so the CPU compute kernels can read from the local copy
// based on the calling thread's TLS NUMA node id.
//
// Available only when the runtime NUMA strategy is GGML_NUMA_STRATEGY_MIRROR.
// When MIRROR is inactive (single-NUMA hosts, or other strategies), the
// allocator falls through to a single-copy allocation that behaves
// identically to the default CPU buffer type. This means the buft can
// be registered as an extra buffer type unconditionally; it only
// replicates when the user opts in via --numa mirror.
ggml_backend_buffer_type_t ggml_backend_cpu_numa_mirror_buffer_type(void);

// Returns true if the given buffer is owned by the mirror buft AND
// currently has two physical copies (i.e. MIRROR was active at alloc
// time). Returns false for any other buffer or for a mirror buffer
// that took the single-copy fallback path.
bool ggml_backend_cpu_numa_mirror_is_mirror(ggml_backend_buffer_t buffer);

// Returns the byte offset from the primary copy (tensor->data on a
// mirror tensor) to the secondary copy. Zero for non-mirror buffers
// and for single-copy fallback. Used by the matmul kernels to compute
// the local-NUMA pointer for the calling thread:
//   const ptrdiff_t alt = ggml_backend_cpu_numa_mirror_alt_offset(t->buffer);
//   const char * data_local = (const char *) t->data + alt * ggml_cpu_get_numa_node();
ptrdiff_t ggml_backend_cpu_numa_mirror_alt_offset(ggml_backend_buffer_t buffer);

// After-op replication hook. Called from ggml_compute_forward by every
// thread after a write op completes. The helper:
//   1. Returns immediately for non-mirror destination buffers (cheap
//      check on the buft alt offset).
//   2. Calls ggml_barrier() to ensure all threads have committed their
//      slice of the op work — most ops slice work by row across threads
//      and do not have an internal barrier, so the calling thread cannot
//      assume the dst tensor is fully written when it returns from the
//      op dispatch.
//   3. The master thread (ith == 0) replicates the dirty region from
//      copy 0 to copy 1; other threads return.
//   4. The threadpool barrier between graph nodes serializes the master
//      sync against the next op, so no extra barrier is needed at the
//      hook exit.
// The set of supported write ops is enumerated in the helper's switch;
// unknown ops fall through to a safe full-tensor memcpy.
void ggml_backend_cpu_numa_mirror_after_op_sync(
        const struct ggml_compute_params * params,
        struct ggml_tensor * tensor);

#ifdef __cplusplus
}
#endif
