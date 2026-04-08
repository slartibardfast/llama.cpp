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

namespace {

// Per-buffer context. Holds both physical bases and the precomputed
// alt offset (= base[1] - base[0]). For the v1 single-copy fallback,
// base[1] is NULL and alt_off is 0; the buffer behaves as a regular
// CPU buffer.
struct mirror_ctx {
    void *    base[2];   // base[0] is the primary copy, base[1] the alt
    size_t    size;
    ptrdiff_t alt_off;   // 0 when single-copy
};

// Buffer alignment matches the default CPU buffer type so set_tensor /
// get_tensor can use plain memcpy without alignment fixups. Reuses the
// global TENSOR_ALIGNMENT macro from ggml-impl.h.

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
    if (ctx->base[1] != nullptr) {
        aligned_free_local(ctx->base[1]);
    }
    aligned_free_local(ctx->base[0]);
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

    // v1: single-copy fallback. The dual-mbind allocator lands in step 3
    // of the Phase 26 #1 implementation. For now, allocate one copy and
    // leave alt_off at 0; the buffer behaves identically to the default
    // CPU buffer type.
    ctx->base[0] = aligned_alloc_local(size);
    ctx->base[1] = nullptr;
    ctx->alt_off = 0;

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

void ggml_backend_cpu_numa_mirror_after_op_sync(struct ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return;
    }
    const ptrdiff_t alt_off = ggml_backend_cpu_numa_mirror_alt_offset(tensor->buffer);
    if (alt_off == 0) {
        // Single-copy fallback or non-mirror buffer; nothing to sync.
        return;
    }

    // The dispatch on tensor->op decides what region to sync. v1 lands
    // the framework with a safe full-tensor fallback; the per-op narrow
    // dirty regions (e.g. set_rows reading the index tensor) are added
    // alongside the dual-mbind allocator in step 4.
    const size_t nbytes = ggml_nbytes(tensor);
    memcpy((char *) tensor->data + alt_off, tensor->data, nbytes);
}
