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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s <model.gguf> [--rw N] [--ctx N] [--type-k NAME] [--verbose]\n",
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
    bool verbose = false;

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
    mparams.n_gpu_layers = 0; /* pure CPU — portable across hosts */

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

    fprintf(stdout, "HARNESS_OK rw=%u ctx=%u type_k=%s rw_type_k=%s decoded=%d\n",
        rw, (uint32_t) llama_n_ctx(ctx), type_k_name.c_str(), rw_type_k_name.c_str(), decoded);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
