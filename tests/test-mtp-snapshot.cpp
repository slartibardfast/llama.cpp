// Plan A foundation test:
//   Verifies that llama_state_seq_get_data_ext / set_data_ext with
//   LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY correctly snapshots and restores
//   the recurrent DeltaNet state of a hybrid model, within the same
//   context and same seq_id.
//
// The flow mirrors what Plan A 1-phase spec decode will do at runtime:
//   1. Process a prompt → state at position P
//   2. Snapshot seq 0 recurrent state → buffer
//   3. Decode N tokens → state at P+N, record sampled token T_ref
//   4. Restore seq 0 from snapshot + seq_rm KV cache past P
//   5. Decode the same N tokens → state at P+N again
//   6. Sample next token, must equal T_ref
//
// Pass criteria: step 6 matches. Failure means the snapshot API
// doesn't faithfully restore the hybrid state and Plan A can't use it.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <vector>

static llama_token greedy(llama_context * ctx) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    llama_token best = 0;
    float best_l = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > best_l) {
            best_l = logits[i];
            best = i;
        }
    }
    return best;
}

int main(int argc, char ** argv) {
    common_params params;
    params.prompt = "Once upon a time, a robot named R2-D2";
    params.n_predict = 8;     // number of tokens to advance between snap and restore
    params.sampling.seed = 42;
    params.sampling.temp = 0.0f;
    params.n_ctx = 4096;
    params.n_parallel = 1;
    params.kv_unified = true;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init_result_ptr init = common_init_from_params(params);
    llama_model   * model = init->model();
    llama_context * ctx   = init->context();

    if (!model || !ctx) {
        fprintf(stderr, "failed to init\n");
        return 1;
    }

    // Tokenize and process prompt
    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, params.prompt, true);
    const int n_prompt = (int) prompt_tokens.size();
    fprintf(stderr, "prompt: %d tokens\n", n_prompt);

    llama_batch batch = llama_batch_init(params.n_ctx, 0, 1);
    common_batch_clear(batch);
    for (int i = 0; i < n_prompt; i++) {
        common_batch_add(batch, prompt_tokens[i], i, {0}, i == n_prompt - 1);
    }
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "prompt decode failed\n");
        return 2;
    }
    int n_past = n_prompt;

    // Phase 2 sanity: print MTP draft stack shape and per-draft top-1
    const float * mtp = llama_get_mtp_logits(ctx);
    const int64_t mtp_v = llama_get_mtp_n_vocab(ctx);
    const int64_t mtp_k = llama_get_mtp_n_drafts(ctx);
    fprintf(stderr, "\nMTP logits shape: vocab=%lld n_drafts=%lld\n", (long long)mtp_v, (long long)mtp_k);
    if (mtp && mtp_v > 0 && mtp_k > 0) {
        for (int64_t j = 0; j < mtp_k; j++) {
            const float * row = mtp + j * mtp_v;
            llama_token best = 0;
            float best_l = row[0];
            for (int64_t i = 1; i < mtp_v; i++) {
                if (row[i] > best_l) { best_l = row[i]; best = (llama_token)i; }
            }
            fprintf(stderr, "  draft[%lld] top-1 token = %d (logit %.3f)\n",
                    (long long)j, best, best_l);
        }
    }

    // Test both full (flags=0) and partial (recurrent-only) snapshots
    for (int pass = 0; pass < 2; pass++) {
        const bool partial = (pass == 1);
        const llama_state_seq_flags flags = partial ? LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY : 0;
        const char * mode_str = partial ? "PARTIAL" : "FULL";

        fprintf(stderr, "\n=== Pass %d: %s snapshot ===\n", pass, mode_str);

        // 1. Snapshot at current position
        size_t snap_size = llama_state_seq_get_size_ext(ctx, 0, flags);
        fprintf(stderr, "%s snapshot size: %zu bytes\n", mode_str, snap_size);
        if (snap_size == 0) {
            fprintf(stderr, "FAIL: snapshot size == 0\n");
            return 3;
        }
        std::vector<uint8_t> snap(snap_size);
        size_t got = llama_state_seq_get_data_ext(ctx, snap.data(), snap.size(), 0, flags);
        if (got != snap_size) {
            fprintf(stderr, "FAIL: snapshot write %zu != %zu\n", got, snap_size);
            return 4;
        }

        const int n_past_at_snap = n_past;

        // 2. Advance by N tokens, record each sampled token
        std::vector<llama_token> first_run_tokens;
        first_run_tokens.reserve(params.n_predict);

        llama_token last_sampled = greedy(ctx);
        for (int i = 0; i < params.n_predict; i++) {
            first_run_tokens.push_back(last_sampled);

            common_batch_clear(batch);
            common_batch_add(batch, last_sampled, n_past, {0}, true);
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "first run decode %d failed\n", i);
                return 5;
            }
            n_past++;
            last_sampled = greedy(ctx);
        }
        const llama_token ref_next = last_sampled;
        fprintf(stderr, "first run advanced %d tokens, ref_next = %d\n",
                params.n_predict, ref_next);
        fprintf(stderr, "  sequence: ");
        for (auto t : first_run_tokens) {
            fprintf(stderr, "%d ", t);
        }
        fprintf(stderr, "\n");

        // 3. Restore the snapshot
        size_t set = llama_state_seq_set_data_ext(ctx, snap.data(), snap.size(), 0, flags);
        if (set != snap_size) {
            fprintf(stderr, "FAIL: restore set %zu != %zu\n", set, snap_size);
            return 6;
        }

        // 4. If partial, also roll back KV cache (since partial only restored recurrent state)
        if (partial) {
            bool ok = llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past_at_snap, -1);
            fprintf(stderr, "partial-only: seq_rm KV from pos %d onward: %s\n",
                    n_past_at_snap, ok ? "ok" : "FAIL");
            if (!ok) {
                fprintf(stderr, "FAIL: partial seq_rm failed\n");
                return 7;
            }
        }
        n_past = n_past_at_snap;

        // 5. Replay the same N tokens and compare
        //    After restore, we need to get the logits at the last prompt position.
        //    Since the snapshot was taken after the prompt decode, the logits buffer
        //    should still be valid (state is at same position). But some restores
        //    invalidate logits — so re-sample by re-decoding one token if needed.
        //    Actually: the snapshot preserves state, but the logits buffer is part
        //    of the context, not the memory. It might still hold the last decode's
        //    logits. Let's just use the recorded first_run_tokens[0] directly.

        llama_token replay_next = first_run_tokens[0];
        for (int i = 0; i < params.n_predict; i++) {
            common_batch_clear(batch);
            common_batch_add(batch, replay_next, n_past, {0}, true);
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "replay decode %d failed\n", i);
                return 8;
            }
            n_past++;
            replay_next = greedy(ctx);

            if (i + 1 < params.n_predict && replay_next != first_run_tokens[i + 1]) {
                fprintf(stderr, "FAIL: replay divergence at step %d: got %d expected %d\n",
                        i + 1, replay_next, first_run_tokens[i + 1]);
                return 9;
            }
        }

        // 6. Final sampled token should match ref
        if (replay_next != ref_next) {
            fprintf(stderr, "FAIL: %s replay next = %d, ref = %d\n", mode_str, replay_next, ref_next);
            return 10;
        }

        fprintf(stderr, "PASS: %s snapshot faithfully restored recurrent+KV state\n", mode_str);
    }

    llama_batch_free(batch);
    fprintf(stderr, "\nALL PASSES ✓\n");
    return 0;
}
