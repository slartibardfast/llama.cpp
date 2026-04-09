// numa-mirror.cpp -- per-NUMA-node replicated CPU buffer type for ggml.
//
// Allocates two physical copies of each buffer (one bound to each NUMA
// node) and exposes a per-buffer alt offset so the CPU compute kernels
// can read from the local copy based on the calling thread's TLS NUMA
// node id. Writes to mirror buffers are replicated by an after-op hook
// in ggml_compute_forward, not by per-op modification.
//
// v1 (this file): scaffolding + single-copy fallback. The dual-mbind
// allocator is wired in step 3 of the Phase 26 #1 implementation. With
// the fallback, behavior is identical to the default CPU buffer type;
// no replication happens. This lets us land the registration plumbing
// independently of the actual mirroring functionality.

#include "numa-mirror.h"

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"

#include <cstdlib>
#include <cstring>
#include <new>

#if defined(__gnu_linux__)
#  include <sys/mman.h>
#  include <numaif.h>
#  include <unistd.h>
#  define GGML_NUMA_MIRROR_HAVE_MBIND 1
#else
#  define GGML_NUMA_MIRROR_HAVE_MBIND 0
#endif

namespace {

// Per-buffer context. Holds both physical bases and the precomputed
// alt offset (= base[1] - base[0]). For the single-copy fallback,
// base[1] is NULL and alt_off is 0; the buffer behaves as a regular
// CPU buffer.
//
// `mbind_path` records whether the bases came from mbind_alloc_on_node
// (true) or posix_memalign (false), so the free path knows which
// deallocator to use.
struct mirror_ctx {
    void *    base[2];   // base[0] is the primary copy, base[1] the alt
    size_t    size;
    ptrdiff_t alt_off;   // 0 when single-copy
    bool      mbind_path;
};

// Buffer alignment matches the default CPU buffer type so set_tensor /
// get_tensor can use plain memcpy without alignment fixups. Reuses the
// global TENSOR_ALIGNMENT macro from ggml-impl.h.

#if GGML_NUMA_MIRROR_HAVE_MBIND

// Round size up to the page boundary so mbind operates on whole pages.
size_t round_up_to_page(size_t size) {
    static const size_t pg = (size_t) sysconf(_SC_PAGESIZE);
    return (size + pg - 1) & ~(pg - 1);
}

// Allocate `size` bytes via mmap and bind the resulting pages to a
// specific NUMA node. Returns the base pointer (page-aligned), or
// MAP_FAILED on error. The mapping is private + anonymous so it costs
// no swap and is freed via munmap.
//
// We use MPOL_BIND so the kernel will fail (rather than fall back to
// other nodes) if `node` runs out of memory. Combined with our 188 GiB
// budget and 96 GiB per node, this is a hard guarantee that the
// allocation lives where we asked for it.
void * mbind_alloc_on_node(size_t size, int node) {
    const size_t aligned_size = round_up_to_page(size);
    void * p = mmap(nullptr, aligned_size,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS,
                    -1, 0);
    if (p == MAP_FAILED) {
        return MAP_FAILED;
    }

    // Build a node mask with just `node` set.
    unsigned long node_mask = 1UL << node;
    const unsigned long max_node = sizeof(node_mask) * 8;

    if (mbind(p, aligned_size, MPOL_BIND, &node_mask, max_node, MPOL_MF_STRICT) != 0) {
        munmap(p, aligned_size);
        return MAP_FAILED;
    }

    return p;
}

void mbind_free(void * p, size_t size) {
    if (p != nullptr && p != MAP_FAILED) {
        munmap(p, round_up_to_page(size));
    }
}

#endif  // GGML_NUMA_MIRROR_HAVE_MBIND

void * aligned_alloc_local(size_t size) {
    void * p = nullptr;
    if (posix_memalign(&p, TENSOR_ALIGNMENT, size) != 0) {
        return nullptr;
    }
    return p;
}

void aligned_free_local(void * p) {
    free(p);
}

const char * mirror_buffer_get_name(ggml_backend_buffer_t /*buffer*/) {
    return "CPU_NUMA_MIRROR";
}

void * mirror_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<mirror_ctx *>(buffer->context);
    return ctx->base[0];
}

void mirror_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<mirror_ctx *>(buffer->context);
#if GGML_NUMA_MIRROR_HAVE_MBIND
    if (ctx->mbind_path) {
        mbind_free(ctx->base[0], ctx->size);
        mbind_free(ctx->base[1], ctx->size);
    } else
#endif
    {
        aligned_free_local(ctx->base[0]);
    }
    delete ctx;
}

void mirror_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                 uint8_t value, size_t offset, size_t size) {
    auto * ctx = static_cast<mirror_ctx *>(buffer->context);
    memset((char *) tensor->data + offset, value, size);
    if (ctx->alt_off != 0) {
        memset((char *) tensor->data + offset + ctx->alt_off, value, size);
    }
}

void mirror_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                              const void * data, size_t offset, size_t size) {
    auto * ctx = static_cast<mirror_ctx *>(buffer->context);
    memcpy((char *) tensor->data + offset, data, size);
    if (ctx->alt_off != 0) {
        memcpy((char *) tensor->data + offset + ctx->alt_off, data, size);
    }
}

void mirror_buffer_get_tensor(ggml_backend_buffer_t /*buffer*/,
                              const struct ggml_tensor * tensor,
                              void * data, size_t offset, size_t size) {
    // Read from the primary copy. The two copies are byte-identical
    // for read-only weights; for KV cache the after-op sync ensures
    // the primary is up to date before any external get_tensor.
    memcpy(data, (const char *) tensor->data + offset, size);
}

bool mirror_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                              const struct ggml_tensor * src,
                              struct ggml_tensor * dst) {
    auto * ctx = static_cast<mirror_ctx *>(buffer->context);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        if (ctx->alt_off != 0) {
            memcpy((char *) dst->data + ctx->alt_off, src->data, ggml_nbytes(src));
        }
        return true;
    }
    return false;
}

void mirror_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = static_cast<mirror_ctx *>(buffer->context);
    memset(ctx->base[0], value, ctx->size);
    if (ctx->base[1] != nullptr) {
        memset(ctx->base[1], value, ctx->size);
    }
}

constexpr ggml_backend_buffer_i mirror_buffer_iface = {
    /* .free_buffer     = */ mirror_buffer_free_buffer,
    /* .get_base        = */ mirror_buffer_get_base,
    /* .init_tensor     = */ nullptr,
    /* .memset_tensor   = */ mirror_buffer_memset_tensor,
    /* .set_tensor      = */ mirror_buffer_set_tensor,
    /* .get_tensor      = */ mirror_buffer_get_tensor,
    /* .cpy_tensor      = */ mirror_buffer_cpy_tensor,
    /* .clear           = */ mirror_buffer_clear,
    /* .reset           = */ nullptr,
};

const char * mirror_buft_get_name(ggml_backend_buffer_type_t /*buft*/) {
    return "CPU_NUMA_MIRROR";
}

ggml_backend_buffer_t mirror_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * ctx = new (std::nothrow) mirror_ctx{};
    if (ctx == nullptr) {
        return nullptr;
    }

    ctx->size = size;

#if GGML_NUMA_MIRROR_HAVE_MBIND
    // Dual-mbind allocator path: bind copy 0 to NUMA node 0 and copy 1
    // to NUMA node 1. The two copies are page-aligned by mmap and the
    // alt offset is the byte distance between them. After load, both
    // copies are populated identically by the buffer's set_tensor /
    // memset_tensor callbacks (which write through to both copies),
    // and the after-op sync hook keeps them in lockstep for any
    // subsequent compute writes.
    void * p0 = mbind_alloc_on_node(size, 0);
    void * p1 = mbind_alloc_on_node(size, 1);
    if (p0 != MAP_FAILED && p1 != MAP_FAILED) {
        ctx->base[0] = p0;
        ctx->base[1] = p1;
        ctx->alt_off = (char *) p1 - (char *) p0;
        ctx->mbind_path = true;
        return ggml_backend_buffer_init(buft, mirror_buffer_iface, ctx, size);
    }
    // mbind failed (probably out of memory on one node, or running on
    // a kernel without NUMA support). Free whatever we got and fall
    // through to the single-copy path so the model still loads, just
    // without replication.
    mbind_free(p0, size);
    mbind_free(p1, size);
#endif

    // Single-copy fallback. Behaves identically to the default CPU buffer
    // type. Used when mbind isn't available or when both nodes can't
    // satisfy the allocation.
    ctx->base[0] = aligned_alloc_local(size);
    ctx->base[1] = nullptr;
    ctx->alt_off = 0;
    ctx->mbind_path = false;

    if (ctx->base[0] == nullptr) {
        delete ctx;
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, mirror_buffer_iface, ctx, size);
}

size_t mirror_buft_get_alignment(ggml_backend_buffer_type_t /*buft*/) {
    return TENSOR_ALIGNMENT;
}

bool mirror_buft_is_host(ggml_backend_buffer_type_t /*buft*/) {
    return true;
}

}  // namespace

ggml_backend_buffer_type_t ggml_backend_cpu_numa_mirror_buffer_type(void) {
    static struct ggml_backend_buffer_type t = {
        /* .iface   = */ {
            /* .get_name         = */ mirror_buft_get_name,
            /* .alloc_buffer     = */ mirror_buft_alloc_buffer,
            /* .get_alignment    = */ mirror_buft_get_alignment,
            /* .get_max_size     = */ nullptr,  // defaults to SIZE_MAX
            /* .get_alloc_size   = */ nullptr,  // defaults to ggml_nbytes
            /* .is_host          = */ mirror_buft_is_host,
        },
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    return &t;
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type_for_runtime(void) {
    // Default callsites (KV cache, recurrent state) ask for the regular
    // CPU buft directly. The helper exists so a future caller can opt
    // into NUMA mirror placement of any buffer that would otherwise go
    // to the regular CPU buft, without making the callsite aware of
    // mirror semantics.
    //
    // For KV/recurrent state, mirroring is OPT-IN via the
    // LLAMA_NUMA_MIRROR_KV env var. Default is OFF because measurement
    // on dual Westmere with Qwen3.5-A3B Q4_K_M showed a ~14% decode
    // regression when KV is on the mirror buffer (hot bench fixture:
    // 8K openclaw fill, --numa mirror -t 12). The cause is per-op sync
    // overhead amortised over many small SET_ROWS writes outweighing
    // the cross-socket savings on KV reads, which are L3-friendly at
    // moderate fill. The code path is kept available because (a) it
    // may pay off at much larger fills where KV scan dominates and
    // (b) it gives us a tested mirror-write framework that any future
    // mirrored buffer (e.g. activations) can re-use.
    if (ggml_cpu_get_numa_strategy() != GGML_NUMA_STRATEGY_MIRROR) {
        return ggml_backend_cpu_buffer_type();
    }
    if (getenv("LLAMA_NUMA_MIRROR_KV") == nullptr) {
        return ggml_backend_cpu_buffer_type();
    }
    return ggml_backend_cpu_numa_mirror_buffer_type();
}

bool ggml_backend_cpu_numa_mirror_is_mirror(ggml_backend_buffer_t buffer) {
    if (buffer == nullptr) {
        return false;
    }
    if (buffer->buft != ggml_backend_cpu_numa_mirror_buffer_type()) {
        return false;
    }
    auto * ctx = static_cast<mirror_ctx *>(buffer->context);
    return ctx->alt_off != 0;
}

ptrdiff_t ggml_backend_cpu_numa_mirror_alt_offset(ggml_backend_buffer_t buffer) {
    if (buffer == nullptr) {
        return 0;
    }
    if (buffer->buft != ggml_backend_cpu_numa_mirror_buffer_type()) {
        return 0;
    }
    auto * ctx = static_cast<mirror_ctx *>(buffer->context);
    return ctx->alt_off;
}

void ggml_backend_cpu_buffer_finalize_load(ggml_backend_buffer_t buffer) {
    if (buffer == nullptr) {
        return;
    }
    if (buffer->buft != ggml_backend_cpu_numa_mirror_buffer_type()) {
        return;
    }
    auto * ctx = static_cast<mirror_ctx *>(buffer->context);
    if (ctx->base[1] == nullptr || ctx->alt_off == 0) {
        return;
    }
    // Bulk replicate the primary copy into the secondary. The model loader
    // calls this once per buffer after all tensors have been loaded via
    // direct cur->data writes (which bypass the buffer set_tensor hook).
    memcpy(ctx->base[1], ctx->base[0], ctx->size);
}

// Per-thread parallel-slice sync. Each thread syncs the dst rows IT wrote
// during the op, matching the per-row work slicing the op kernels use.
// Because each thread reads only what it itself just wrote, no barrier is
// needed: there is no race with other threads' writes.
//
// This avoids the cost of a master-thread serial sync + cross-socket
// barrier on every mirror-write op, which on dual-socket Westmere costs
// many microseconds per op and at ~120 mirror-write ops per token adds
// up to a measurable decode regression.
//
// Slicing convention: rows-of-the-write-region split equally across
// threads, exactly matching the (nr + nth - 1) / nth / ir0 / ir1 pattern
// in compute_forward_set_rows_f32 and compute_forward_dup_bytes.

// SET_ROWS slice sync. The op iterates rows 0..nr of src0; this thread
// owns rows ir0..ir1 (matching the op kernel's slicing). For each owned
// row, look up the destination row index in src1 and sync that dst row.
//
// For Qwen3.5-A3B at decode the typical SET_ROWS into the K cache writes
// 1 row of ~24 bytes (TQ_KV_1B) per call; the V cache writes 1 row of
// ~66 bytes (TQ_V_4B). With nr=1 only thread 0 owns work; the others
// run with empty ir0..ir1 and exit immediately with no memcpy.
static void mirror_sync_set_rows_slice(struct ggml_tensor * tensor, ptrdiff_t alt_off,
                                        int ith, int nth) {
    const ggml_tensor * src0 = tensor->src[0];
    const ggml_tensor * src1 = tensor->src[1];
    if (src0 == nullptr || src1 == nullptr) {
        return;  // defensive
    }

    const int64_t nc   = src0->ne[0];
    const int64_t nr   = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const size_t  nb10 = src1->nb[0];
    const size_t  nb11 = src1->nb[1];
    const size_t  nb12 = src1->nb[2];
    const size_t  nb1  = tensor->nb[1];
    const size_t  nb2  = tensor->nb[2];
    const size_t  nb3  = tensor->nb[3];

    const int64_t dr  = (nr + nth - 1) / nth;
    const int64_t ir0 = dr * ith;
    const int64_t ir1 = (ir0 + dr) < nr ? (ir0 + dr) : nr;

    if (ir0 >= ir1) {
        return;  // empty slice for this thread
    }

    const size_t row_size = ggml_row_size(tensor->type, nc);
    const bool   idx_i64  = (src1->type == GGML_TYPE_I64);

    for (int64_t i03 = 0; i03 < ne03; ++i03) {
        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            for (int64_t i = ir0; i < ir1; ++i) {
                const int64_t i12 = i03 % ne12;
                const int64_t i11 = i02 % ne11;
                const int64_t i10 = i;

                const char * idx_ptr = (const char *) src1->data
                                     + i10*nb10 + i11*nb11 + i12*nb12;
                const int64_t i1 = idx_i64
                                 ? *(const int64_t *) idx_ptr
                                 : (int64_t) *(const int32_t *) idx_ptr;

                char * dst_row = (char *) tensor->data
                               + i1*nb1 + i02*nb2 + i03*nb3;
                memcpy(dst_row + alt_off, dst_row, row_size);
            }
        }
    }
}

// Per-thread row-slice sync of the entire dst tensor. Used for CPY/DUP
// and for ops that write the full dst extent. Each thread copies its
// slice of dst rows to the alt copy. No barrier needed because each
// thread reads only the rows it wrote.
static void mirror_sync_rows_slice(struct ggml_tensor * tensor, ptrdiff_t alt_off,
                                    int ith, int nth) {
    const int64_t nr = ggml_nrows(tensor);

    if (nr <= 1) {
        // Single-row or zero-row dst: master thread does the whole copy.
        // The op kernel similarly funnels into a single thread for nr<=1.
        if (ith == 0) {
            const size_t nbytes = ggml_nbytes(tensor);
            memcpy((char *) tensor->data + alt_off, tensor->data, nbytes);
        }
        return;
    }

    const int64_t dr  = (nr + nth - 1) / nth;
    const int64_t ir0 = dr * ith;
    const int64_t ir1 = (ir0 + dr) < nr ? (ir0 + dr) : nr;

    if (ir0 >= ir1) {
        return;  // empty slice for this thread
    }

    // Treat dst as a flat row-major buffer with row stride nb[1] (which
    // is the byte stride to advance one row in dim 1). This matches what
    // ggml_nrows iterates over and what the CPY/DUP kernels write.
    const size_t row_bytes = tensor->nb[1];
    char * dst_base = (char *) tensor->data + ir0 * row_bytes;
    const size_t span = (size_t)(ir1 - ir0) * row_bytes;
    memcpy(dst_base + alt_off, dst_base, span);
}

void ggml_backend_cpu_numa_mirror_after_op_sync(
        const struct ggml_compute_params * params,
        struct ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return;
    }
    const ptrdiff_t alt_off = ggml_backend_cpu_numa_mirror_alt_offset(tensor->buffer);
    if (alt_off == 0) {
        // Single-copy fallback or non-mirror buffer; nothing to sync.
        // Cheap early return — most tensors in a graph live in compute
        // scratch (regular CPU buft, alt_off == 0) so this is the
        // common case and runs on every thread.
        return;
    }

    const int ith = params->ith;
    const int nth = params->nth;

    switch (tensor->op) {
        case GGML_OP_SET_ROWS:
            // Critical case: KV cache writes go through SET_ROWS. Each
            // thread syncs the dst rows it wrote (looked up via src[1]).
            mirror_sync_set_rows_slice(tensor, alt_off, ith, nth);
            break;

        case GGML_OP_CPY:
        case GGML_OP_DUP:
            // Recurrent state checkpointing and similar. The op slices
            // dst by rows; sync the same row slice.
            mirror_sync_rows_slice(tensor, alt_off, ith, nth);
            break;

        case GGML_OP_SCALE:
            // In-place SCALE on a mirrored tensor. Same row slicing.
            mirror_sync_rows_slice(tensor, alt_off, ith, nth);
            break;

        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
        case GGML_OP_GATED_DELTA_NET:
            // SSM/GDN ops typically write to a fresh compute-scratch dst
            // (alt_off == 0, early-exit above). This case only fires when
            // the dst happens to be on a mirror buffer; row-slice sync.
            mirror_sync_rows_slice(tensor, alt_off, ith, nth);
            break;

        default: {
            // Safety net: an unhandled write op falls back to a master
            // sync with explicit barrier. The barrier is needed because
            // we don't know how this op sliced its work across threads.
            // Log once so we know to add a narrow case if this fires.
            static enum ggml_op s_warned_op = GGML_OP_NONE;
            if (ith == 0 && s_warned_op != tensor->op) {
                s_warned_op = tensor->op;
                fprintf(stderr,
                        "ggml-cpu numa mirror: unhandled write op %s (%s) "
                        "on mirror buffer — falling back to full-tensor sync\n",
                        ggml_op_name(tensor->op), tensor->name);
            }
            ggml_barrier(params->threadpool);
            if (ith == 0) {
                const size_t nbytes = ggml_nbytes(tensor);
                memcpy((char *) tensor->data + alt_off, tensor->data, nbytes);
            }
            break;
        }
    }
}
