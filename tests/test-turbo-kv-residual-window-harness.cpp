/*
 * test-turbo-kv-residual-window-harness.cpp
 *
 * Minimal context-init harness for the residual-window feature.
 * Purpose: exercise llama_context construction with varied
 * residual_window / cache_type_k combinations and emit a single
 * structured status line to stdout. No decoding, no conversation
 * mode — so the MTP-2PHASE debug stream that floods stderr during
 * normal llama-cli use does not interfere with log inspection.
 *
 * Usage:
 *   test-turbo-kv-residual-window-harness <model.gguf> [options]
 *
 * Options (positional, all optional):
 *   --rw N            residual_window value (default 0)
 *   --ctx N           n_ctx (default 512)
 *   --type-k NAME     K cache type: f16 | f32 | turbo_kv_4b (default f16)
 *   --rw-type-k NAME  residual-window overlay dtype: auto | f16 | bf16
 *                     (default auto — inherit from model's native K dtype)
 *   --append N        run N incremental decode() calls with a dummy
 *                     token each to exercise the KV-cache write path
 *                     (default 0 — skip decode, init-only smoke)
 *   --check-window    after --append, peek every overlay slot and assert
 *                     that exactly min(N, rw) slots per stream are non-zero
 *                     per active layer. Emits WINDOW_CHECK_{PASS,FAIL}
 *                     lines to stdout. Exit 5 on mismatch.
 *   --verbose         enable llama info logging to stderr
 *
 * Exit codes:
 *   0  context created and freed cleanly
 *   1  argument error
 *   2  model load failed
 *   3  context init failed
 *
 * Final stdout line (on exit 0):
 *   HARNESS_OK rw=N ctx=M type_k=NAME
 *
 * stderr (when --verbose) contains the llama_kv_cache init log
 * lines including "KV buffer size = X MiB" — the observable proof
 * that the fp16 side-buffer allocation happened when rw > 0 and
 * type_k is a TURBO_KV type.
 */

#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s <model.gguf> [--rw N] [--ctx N] [--type-k NAME] [--ngl N] [--verbose]\n",
        argv0);
}

static ggml_type parse_type_k(const char * name) {
    if (strcmp(name, "f16")         == 0) { return GGML_TYPE_F16; }
    if (strcmp(name, "f32")         == 0) { return GGML_TYPE_F32; }
    if (strcmp(name, "turbo_kv_4b") == 0) { return GGML_TYPE_TURBO_KV_4B; }
    return GGML_TYPE_COUNT; /* signal: unknown */
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    uint32_t  rw             = 0;
    uint32_t  n_ctx          = 512;
    ggml_type type_k         = GGML_TYPE_F16;
    std::string type_k_name  = "f16";
    ggml_type rw_type_k      = GGML_TYPE_COUNT; // auto
    std::string rw_type_k_name = "auto";
    int append_n = 0;
    bool check_window = false;
    bool verbose = false;
    int seq_rm_then_peek_p_min = -1;
    int seq_rm_then_peek_p_max = -1;
    int n_gpu_layers = 0; /* default CPU; --ngl pushes to active GPU backend */

    for (int i = 2; i < argc; ++i) {
        const char * a = argv[i];
        if (strcmp(a, "--rw") == 0 && i + 1 < argc) {
            int v = atoi(argv[++i]);
            if (v < 0) { fprintf(stderr, "--rw must be >= 0\n"); return 1; }
            rw = (uint32_t) v;
        } else if (strcmp(a, "--ctx") == 0 && i + 1 < argc) {
            int v = atoi(argv[++i]);
            if (v <= 0) { fprintf(stderr, "--ctx must be > 0\n"); return 1; }
            n_ctx = (uint32_t) v;
        } else if (strcmp(a, "--type-k") == 0 && i + 1 < argc) {
            type_k_name = argv[++i];
            type_k = parse_type_k(type_k_name.c_str());
            if (type_k == GGML_TYPE_COUNT) {
                fprintf(stderr, "unknown --type-k value: %s\n", type_k_name.c_str());
                return 1;
            }
        } else if (strcmp(a, "--rw-type-k") == 0 && i + 1 < argc) {
            rw_type_k_name = argv[++i];
            if (rw_type_k_name == "auto") {
                rw_type_k = GGML_TYPE_COUNT;
            } else if (rw_type_k_name == "f16") {
                rw_type_k = GGML_TYPE_F16;
            } else if (rw_type_k_name == "bf16") {
                rw_type_k = GGML_TYPE_BF16;
            } else {
                fprintf(stderr, "unknown --rw-type-k value: %s (auto|f16|bf16)\n", rw_type_k_name.c_str());
                return 1;
            }
        } else if (strcmp(a, "--append") == 0 && i + 1 < argc) {
            int v = atoi(argv[++i]);
            if (v < 0) { fprintf(stderr, "--append must be >= 0\n"); return 1; }
            append_n = v;
        } else if (strcmp(a, "--check-window") == 0) {
            check_window = true;
        } else if (strcmp(a, "--seq-rm-then-peek") == 0 && i + 2 < argc) {
            seq_rm_then_peek_p_min = atoi(argv[++i]);
            seq_rm_then_peek_p_max = atoi(argv[++i]);
            if (seq_rm_then_peek_p_min < 0 || seq_rm_then_peek_p_max < seq_rm_then_peek_p_min) {
                fprintf(stderr, "--seq-rm-then-peek expects 0 <= p_min <= p_max\n");
                return 1;
            }
        } else if (strcmp(a, "--ngl") == 0 && i + 1 < argc) {
            int v = atoi(argv[++i]);
            if (v < 0) { fprintf(stderr, "--ngl must be >= 0\n"); return 1; }
            n_gpu_layers = v;
        } else if (strcmp(a, "--verbose") == 0) {
            verbose = true;
        } else {
            fprintf(stderr, "unknown arg: %s\n", a);
            print_usage(argv[0]);
            return 1;
        }
    }

    /* Quiet mode by default: suppress llama logs to keep stdout clean
     * for grep-based assertions in test drivers. --verbose restores
     * the default (info-to-stderr) logger. */
    if (!verbose) {
        llama_log_set(
            [](ggml_log_level, const char *, void *) {},
            nullptr);
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers; /* default 0 = pure CPU; --ngl probes a GPU backend */

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "model load failed: %s\n", model_path);
        llama_backend_free();
        return 2;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                  = n_ctx;
    cparams.n_batch                = n_ctx;      /* safe default for PP */
    cparams.n_ubatch               = n_ctx;
    cparams.type_k                 = type_k;
    cparams.residual_window        = rw;
    cparams.residual_window_type_k = rw_type_k;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "context init failed\n");
        llama_model_free(model);
        llama_backend_free();
        return 3;
    }

    int decoded = 0;
    if (append_n > 0) {
        // Drive the KV-cache write path with append_n dummy tokens.
        // Token id 0 is typically <unk> / BOS-ish but always exists in
        // the vocabulary; we don't care about the logits — just that
        // decode() commits to the cache without asserting.
        llama_batch batch = llama_batch_init(1, /*embd=*/ 0, /*n_seq_max=*/ 1);
        for (int i = 0; i < append_n; ++i) {
            batch.n_tokens    = 1;
            batch.token[0]    = 0;
            batch.pos[0]      = i;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0]   = 0;

            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "decode failed at token %d\n", i);
                llama_batch_free(batch);
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 4;
            }
            decoded++;
        }
        llama_batch_free(batch);
    }

    // --check-window: peek every overlay slot after --append. A slot is
    // "touched" if any byte is non-zero; the expected count per (layer,
    // stream) is min(append_n, rw). Iterates a generous layer range
    // (nlayer up to the model's report) and skips layers with no overlay
    // (peek returns 0 bytes).
    int check_status = 0;
    if (check_window && rw > 0) {
        const int n_layer = llama_model_n_layer(model);
        llama_memory_t mem = llama_get_memory(ctx);
        const int expected = (append_n >= (int) rw) ? (int) rw : append_n;

        int layers_checked = 0;
        int layers_ok      = 0;
        int layers_fail    = 0;

        // One slot can be at most a few KiB on wide models; 65536 bytes
        // is a comfortable upper bound for any realistic head dim.
        std::vector<uint8_t> buf(65536);

        for (int il = 0; il < n_layer; ++il) {
            size_t slot_nbytes = llama_memory_residual_window_slot_nbytes(mem, il);
            if (slot_nbytes == 0) {
                continue; // no overlay on this layer
            }
            if (slot_nbytes > buf.size()) {
                buf.resize(slot_nbytes);
            }
            layers_checked++;

            const int stream = 0; // unified cache; harness uses n_seq_max=1
            int touched = 0;
            for (uint32_t s = 0; s < rw; ++s) {
                size_t got = llama_memory_residual_window_peek(
                        mem, il, stream, (int) s, buf.data(), slot_nbytes);
                if (got != slot_nbytes) {
                    fprintf(stderr, "WINDOW_CHECK il=%d stream=%d slot=%u: peek size mismatch (%zu != %zu)\n",
                        il, stream, s, got, slot_nbytes);
                    layers_fail++;
                    check_status = 5;
                    break;
                }
                bool nonzero = false;
                for (size_t b = 0; b < got; ++b) {
                    if (buf[b] != 0) { nonzero = true; break; }
                }
                if (nonzero) touched++;
            }

            if (touched == expected) {
                layers_ok++;
                if (verbose) {
                    fprintf(stdout, "WINDOW_CHECK_PASS il=%d touched=%d expected=%d slot_bytes=%zu\n",
                        il, touched, expected, slot_nbytes);
                }
            } else {
                fprintf(stdout, "WINDOW_CHECK_FAIL il=%d touched=%d expected=%d slot_bytes=%zu\n",
                    il, touched, expected, slot_nbytes);
                layers_fail++;
                check_status = 5;
            }
        }

        fprintf(stdout, "WINDOW_CHECK layers=%d ok=%d fail=%d expected_per_layer=%d\n",
            layers_checked, layers_ok, layers_fail, expected);
    }

    // --seq-rm-then-peek: snapshot every overlay slot, call seq_rm on the
    // requested position range, peek each slot again, and report how many
    // slots changed. With ReconcileOverlayOnSequenceRemoval implemented and
    // an f16 main cache (where the f16 → f32 → f16 reconciliation round-trip
    // is lossless), the expected count of changed slots equals the number of
    // ring slots whose pre-removal last writer was in the removed range:
    //     |{ p % rw : p in [p_min, p_max] }|
    // Without reconciliation, ZERO slots would change. This is the binary
    // gate for the rewind hazard fix.
    //
    // KNOWN LIMITATION: on hybrid models (Qwen3.5 et al.), the public
    // llama_memory_seq_rm routes through llama_memory_hybrid::seq_rm,
    // which fails when the recurrent cache cannot roll back the SSM
    // state without a checkpoint. The result is changed=0 because
    // seq_rm returned false before reaching the attention cache. To
    // exercise reconcile_overlay_after_removal directly, this test
    // needs either (a) a non-hybrid model with rw>0, or (b) a test-
    // only API that bypasses the hybrid wrapper. The reconcile path
    // itself is verified by code review until that test infrastructure
    // lands.
    if (seq_rm_then_peek_p_min >= 0 && rw > 0) {
        const int n_layer = llama_model_n_layer(model);
        llama_memory_t mem = llama_get_memory(ctx);
        const int p_min = seq_rm_then_peek_p_min;
        const int p_max = seq_rm_then_peek_p_max;

        // Pick a layer with an overlay to inspect.
        int probe_il = -1;
        size_t probe_slot_nbytes = 0;
        for (int il = 0; il < n_layer; ++il) {
            size_t nb = llama_memory_residual_window_slot_nbytes(mem, il);
            if (nb > 0) { probe_il = il; probe_slot_nbytes = nb; break; }
        }
        if (probe_il < 0) {
            fprintf(stdout, "SEQ_RM_PEEK_SKIP no_overlay_layers\n");
        } else {
            std::vector<uint8_t> before(rw * probe_slot_nbytes);
            std::vector<uint8_t> after(rw * probe_slot_nbytes);

            for (uint32_t s = 0; s < rw; ++s) {
                size_t got = llama_memory_residual_window_peek(
                        mem, probe_il, /*stream=*/0, (int) s,
                        before.data() + s * probe_slot_nbytes, probe_slot_nbytes);
                if (got != probe_slot_nbytes) {
                    fprintf(stderr, "SEQ_RM_PEEK pre-snapshot peek mismatch slot=%u\n", s);
                    check_status = 6;
                    break;
                }
            }

            // Apply seq_rm via the public API. seq_id 0 (the only sequence
            // the harness uses). Use [p_min, -1) — truncate from p_min
            // onwards — to match MTP rejection semantics and also because
            // hybrid models reject partial-range seq_rm (the recurrent
            // cache only supports tail-truncate).
            const bool ok = llama_memory_seq_rm(mem, /*seq_id=*/0, p_min, -1);
            if (!ok) {
                fprintf(stderr, "SEQ_RM_PEEK seq_rm returned false (likely hybrid recurrent rejection)\n");
                check_status = 6;
            }

            for (uint32_t s = 0; s < rw; ++s) {
                size_t got = llama_memory_residual_window_peek(
                        mem, probe_il, /*stream=*/0, (int) s,
                        after.data() + s * probe_slot_nbytes, probe_slot_nbytes);
                if (got != probe_slot_nbytes) {
                    fprintf(stderr, "SEQ_RM_PEEK post-snapshot peek mismatch slot=%u\n", s);
                    check_status = 6;
                    break;
                }
            }

            int changed = 0;
            for (uint32_t s = 0; s < rw; ++s) {
                if (std::memcmp(before.data() + s * probe_slot_nbytes,
                                after.data()  + s * probe_slot_nbytes,
                                probe_slot_nbytes) != 0) {
                    changed++;
                }
            }

            // Expected changed slots: { p % rw : p in [p_min, p_max] }
            // For hybrid models we used tail-truncate (p1=-1), so the
            // effective removed range is [p_min, append_n - 1] (= the
            // tail of what was decoded). Use the requested p_max as a
            // hard upper bound but cap at append_n - 1 to be safe.
            const int p_max_eff = (p_max < append_n - 1) ? p_max : (append_n - 1);
            std::vector<bool> expected(rw, false);
            for (int p = p_min; p <= p_max_eff; ++p) {
                expected[(uint32_t)(((p % (int) rw) + (int) rw) % (int) rw)] = true;
            }
            int n_expected = 0;
            for (uint32_t s = 0; s < rw; ++s) {
                if (expected[s]) n_expected++;
            }

            fprintf(stdout, "SEQ_RM_PEEK il=%d p_min=%d p_max=%d changed=%d expected=%d\n",
                    probe_il, p_min, p_max, changed, n_expected);
            if (changed != n_expected) {
                check_status = 6;
            }
        }
    }

    fprintf(stdout, "HARNESS_OK rw=%u ctx=%u type_k=%s rw_type_k=%s decoded=%d\n",
        rw, (uint32_t) llama_n_ctx(ctx), type_k_name.c_str(), rw_type_k_name.c_str(), decoded);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return check_status;
}
