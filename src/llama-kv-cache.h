#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"

#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

//
// llama_kv_cache
//

class llama_kv_cache : public llama_memory_i {
public:
    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    llama_kv_cache(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_k_static, // split K: static dims type. GGML_TYPE_COUNT = no split
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
                     uint32_t   residual_window, // fp16 rolling-tail size for TURBO_KV caches (0 = disabled)
                    ggml_type   residual_window_type_k, // overlay dtype (F16 or BF16). Ignored when residual_window == 0.
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse);

    ~llama_kv_cache() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache specific API
    //

    uint32_t get_size()     const;
    uint32_t get_n_stream() const;

    bool get_has_shift() const;

    ggml_type type_k() const;
    ggml_type type_k_static() const;
    ggml_type type_v() const;

    bool is_split_k() const;

    //
    // graph_build API
    //

    uint32_t get_n_kv(const slot_info & sinfo) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;

    // split K: get views of rope and static portions
    ggml_tensor * get_k_rope  (ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_k_static(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    // split K: store rope and static portions separately
    ggml_tensor * cpy_k_rope  (ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_k_static(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;

    // fp16 rolling-tail write: additionally store k_cur as fp16 into the
    // per-layer window buffer at the slots given by k_window_idxs. Only
    // meaningful when has_residual_window(); returns nullptr otherwise.
    ggml_tensor * cpy_k_window(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_window_idxs, int32_t il, const slot_info & sinfo) const;

    // Residual-window two-pass read-path K/V views for Pass B. See the
    // equivalent llama_kv_cache_context methods for semantics. The
    // window_reorder / v_window_idxs tensors are built by the graph
    // builder at input-build time and populated at set_input time.
    ggml_tensor * get_k_window(ggml_context * ctx, int32_t il,
                               ggml_tensor * window_reorder,
                               const slot_info & sinfo) const;
    ggml_tensor * get_v_window(ggml_context * ctx, int32_t il, uint32_t n_kv,
                               ggml_tensor * v_window_idxs,
                               const slot_info & sinfo) const;

    // Populate the per-ubatch index tensors for the overlay-K reorder
    // and the main-V-cache slice. v_window_idxs takes sinfo because the
    // current ubatch's just-written positions land at slots given by
    // sinfo.idxs[stream][token].
    void set_input_window_reorder(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_window_idxs (ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    // whether the cache was constructed with residual_window > 0 (and
    // therefore has the per-layer k_window_fp16 tensors allocated)
    bool has_residual_window() const;
    uint32_t get_residual_window() const;

    // test-facing introspection (see llama_memory_i for contract)
    size_t peek_k_window_slot(int32_t il, int32_t stream, int32_t slot,
                              void * dst, size_t dst_size) const override;
    size_t get_k_window_slot_nbytes(int32_t il) const override;

    // ReconcileOverlayOnSequenceRemoval (turbo_kv_residual_window.allium):
    // restore-from-main strategy. Called by seq_rm after the cell-removal
    // loop. For each ring slot in [0, rw), find the highest-position
    // surviving cell whose pos % rw == s and rewrite that slot from
    // main-cache K (dequant + dtype-convert as needed). Slots with no
    // surviving writer are zeroed. No-op when residual_window == 0.
    void reconcile_overlay_after_removal(uint32_t stream_idx);

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    bool update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    // I64 [n_tokens] tensor of fp16-window slot indices per token. Only
    // built when has_residual_window(); returns nullptr otherwise.
    ggml_tensor * build_input_k_window_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    // I32 [rw, n_stream] index tensors for the two-pass read path's
    // Pass B. Allocated when has_residual_window(); nullptr otherwise.
    ggml_tensor * build_input_window_reorder(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_window_idxs (ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    // Populate k_window_idxs with per-token slot positions: for each
    // token i, data[i] = s*residual_window + (ubatch.pos[i] %
    // residual_window) where s is the stream index. No-op when
    // has_residual_window() is false.
    void set_input_k_window_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;

    // Residual-window two-pass read-path masks. Shape [n_kv, n_tps, 1,
    // n_stream], identical to the base kq_mask. Only valid when
    // has_residual_window(); caller must guard.
    //   pass_a: main-cache attending to positions OUTSIDE the last
    //           residual_window (old tail). Cells with stored position
    //           p0 > p1 - residual_window are -inf in addition to the
    //           base causal/seq mask.
    //   pass_b: main-cache attending to positions INSIDE the last
    //           residual_window (recent head). Cells with stored
    //           position p0 <= p1 - residual_window are -inf.
    // Together the two masks partition the base kq_mask (each visible
    // cell is unmasked in exactly one of the two passes, masked-out
    // cells are -inf in both).
    void set_input_kq_mask_pass_a(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_kq_mask_pass_b(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;

    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

    // Pre-RoPE K: populate per-cell absolute positions for on-the-fly RoPE
    void set_input_k_pos(ggml_tensor * dst, const slot_info & sinfo) const;

    // Pre-RoPE Pass-B overlay K positions: populate [rw * 4] tensor with
    // the absolute positions of the rw recent-window slots in position
    // order (max_pos - rw + 1 + s for s in [0, rw)).
    void set_input_window_k_pos(ggml_tensor * dst, const llama_ubatch * ubatch) const;

private:
    const llama_model & model;
    const llama_hparams & hparams;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        // split K: separate tensors for RoPE and static dims
        ggml_tensor * k_rope   = nullptr; // [n_rot * n_head_kv, kv_size, n_stream]
        ggml_tensor * k_static = nullptr; // [(head_dim - n_rot) * n_head_kv, kv_size, n_stream]

        // fp16 rolling-tail buffer for TURBO_KV residual-window caches.
        // Shape: [n_embd_k_gqa, residual_window, n_stream]. Nullptr when
        // residual_window == 0 on the owning cache. Allocated as GGML_TYPE_F16
        // regardless of the K type — quantisation only applies to positions
        // that have been evicted from this buffer. See
        // turbo_kv_residual_window.allium for semantics.
        ggml_tensor * k_window_fp16 = nullptr;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
    };

    bool v_trans = true;  // the value tensor is transposed

    // split K cache types (GGML_TYPE_COUNT means no split)
    ggml_type type_k_static_ = GGML_TYPE_COUNT;

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // required padding
    const uint32_t n_pad = 1;

    // SWA
    const uint32_t n_swa = 0;

    // residual window: when > 0, the last N positions per stream are kept in
    // fp16 in k_window_fp16 buffers rather than being quantised into k/k_rope.
    // Only meaningful for TURBO_KV_* K types; ignored by fp16/fp32 K caches.
    const uint32_t residual_window = 0;

    // Overlay dtype — F16 or BF16. Resolved at context init from the user's
    // cparam (which may be GGML_TYPE_COUNT = auto). Governs both the
    // allocation type of k_window_fp16 and the cast inside cpy_k_window.
    const ggml_type residual_window_type_k = GGML_TYPE_F16;

    // env: LLAMA_ATTN_ROT_DISABLE
    bool attn_rot_k = false;
    bool attn_rot_v = false;

    // if all layers participating in the cache have constant head size, the value is stored here
    // otherwise the value is -1
    int32_t n_embd_head_k_all = 0;
    int32_t n_embd_head_v_all = 0;

    // pre-computed hadamard martrices
    std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    std::vector<llama_kv_cells> v_cells;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    std::vector<kv_layer> layers;

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * rot,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale,
                       uint32_t   il) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);
};

class llama_kv_cache_context : public llama_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    // used for errors
    llama_kv_cache_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_context(
            llama_kv_cache * kv);

    // used to create an update context
    llama_kv_cache_context(
            llama_kv_cache * kv,
            llama_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    llama_kv_cache_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;

    ggml_type type_k() const;
    ggml_type type_k_static() const;
    ggml_type type_v() const;

    bool is_split_k() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    // split K: get views of rope and static portions
    ggml_tensor * get_k_rope  (ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_k_static(ggml_context * ctx, int32_t il) const;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be laid out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    // split K: store rope and static portions separately
    ggml_tensor * cpy_k_rope  (ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_k_static(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;

    // fp16 rolling-tail overlay write. Only meaningful when
    // has_residual_window(); returns nullptr otherwise so the caller
    // can guard with a simple null-check.
    ggml_tensor * cpy_k_window(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_window_idxs, int32_t il) const;

    // Residual-window two-pass read-path views for Pass B. Both return
    // tensors shaped [DK|DV, n_head_kv, rw, n_stream] in POSITION ORDER
    // (oldest-of-window first). Requires single-stream, sequentially-
    // stored cache — caller must guard.
    //
    //   get_k_window: reorders the ring-buffer overlay slots into
    //       position order via ggml_get_rows(overlay, window_reorder).
    //   get_v_window: picks the last rw positions out of the main V
    //       cache via ggml_get_rows(v, v_window_idxs).
    //
    // Returns nullptr when the cache has no overlay or this layer has
    // no K slot (recurrent layer in a hybrid cache).
    ggml_tensor * get_k_window(ggml_context * ctx, int32_t il,
                               ggml_tensor * window_reorder) const;
    ggml_tensor * get_v_window(ggml_context * ctx, int32_t il,
                               ggml_tensor * v_window_idxs) const;

    // Per-ubatch I32 index tensors of shape [rw, n_stream] driving
    // get_k_window / get_v_window. See the llama_kv_cache method
    // comments for semantics.
    ggml_tensor * build_input_window_reorder(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_window_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    void set_input_window_reorder(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_window_idxs (ggml_tensor * dst, const llama_ubatch * ubatch) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_k_window_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    bool     has_residual_window() const;
    uint32_t get_residual_window() const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    // Pre-RoPE K storage: per-cell absolute position indices for on-the-fly RoPE.
    // Returns a [n_kv] I32 tensor, or nullptr if K type is not pre-RoPE.
    ggml_tensor * build_input_k_pos(ggml_context * ctx) const;

    // Pre-RoPE Pass-B overlay K positions: [rw * 4] I32 tensor populated in
    // position order with the absolute position of each window slot
    // (max_pos - rw + 1 + s). nullptr when overlay is disabled or the cache
    // type stores post-RoPE K (i.e. self_k_pos is also nullptr).
    ggml_tensor * build_input_window_k_pos(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_k_window_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;

    // Residual-window two-pass masks (see llama_kv_cache).
    void set_input_kq_mask_pass_a(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_kq_mask_pass_b(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;

    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

    void set_input_k_pos(ggml_tensor * dst) const;
    void set_input_window_k_pos(ggml_tensor * dst, const llama_ubatch * ubatch) const;

private:
    llama_memory_status status;

    llama_kv_cache * kv;
    llama_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;
};
