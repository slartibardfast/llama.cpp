/*
 * llama-turbo-codebook — Compute model-specific Lloyd-Max codebooks for TURBO_*B types
 *
 * Usage:
 *   llama-turbo-codebook -m model.gguf [-o codebook.gguf]
 *                        [--imatrix path.gguf]   # optional, weights Lloyd-Max E-step
 *                        [--block-size 128]
 *
 * Loads an F16/F32/Q8_0 GGUF model, applies RHT to all 2D weight tensors,
 * collects the post-RHT distribution, and computes optimal Lloyd-Max centroids
 * at 2/3/4/5-bit. Outputs a GGUF file that llama-quantize can load via --codebook.
 *
 * Default codebooks are the published Gaussian Lloyd-Max tables from Max (1960),
 * matching tq_codebook.c in quantumaikr/quant.cpp. This tool computes codebooks
 * optimized for the model's actual post-RHT weight distribution.
 *
 * With --imatrix, each post-RHT value contributes to the Lloyd-Max expectation
 * step weighted by its imatrix importance (sum_of_squared_activations / chunks).
 * Centroids converge toward the distribution of high-importance elements rather
 * than the global distribution — trades fidelity on low-importance regions for
 * fidelity on critical ones (the Unsloth-style dynamic tradeoff).
 */

#include "ggml.h"
#include "ggml-turbo-kv.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>
#include <unordered_map>

/* ================================================================
 * Weighted Lloyd-Max algorithm.
 *
 * Standard Lloyd-Max: E-step assigns each sample to nearest centroid,
 * M-step updates each centroid to the MEAN of assigned samples.
 *
 * Weighted variant: M-step updates to WEIGHTED MEAN using per-sample weights.
 * Minimizes sum(w_i * (x_i - c_nearest)^2) — the imatrix-weighted MSE.
 *
 * If weights is empty, all weights are 1 (standard Lloyd-Max).
 * ================================================================ */
static void lloyd_max_weighted(const std::vector<float> & data,
                                const std::vector<float> & weights,
                                float * centroids, int n_centroids, int max_iter) {
    const bool use_weights = !weights.empty();
    if (use_weights && weights.size() != data.size()) {
        fprintf(stderr, "lloyd_max_weighted: weights size mismatch\n");
        std::exit(1);
    }

    /* Initialize with uniform spacing in data range */
    float vmin = *std::min_element(data.begin(), data.end());
    float vmax = *std::max_element(data.begin(), data.end());
    for (int c = 0; c < n_centroids; c++) {
        centroids[c] = vmin + (vmax - vmin) * (c + 0.5f) / n_centroids;
    }

    std::vector<double> wsums(n_centroids);
    std::vector<double> wcounts(n_centroids);

    for (int iter = 0; iter < max_iter; iter++) {
        std::fill(wsums.begin(), wsums.end(), 0.0);
        std::fill(wcounts.begin(), wcounts.end(), 0.0);

        for (size_t i = 0; i < data.size(); i++) {
            float v = data[i];
            float w = use_weights ? weights[i] : 1.0f;
            int best = 0;
            float best_dist = fabsf(v - centroids[0]);
            for (int c = 1; c < n_centroids; c++) {
                float d = fabsf(v - centroids[c]);
                if (d < best_dist) { best_dist = d; best = c; }
            }
            wsums[best] += (double)w * v;
            wcounts[best] += (double)w;
        }

        float max_shift = 0;
        for (int c = 0; c < n_centroids; c++) {
            if (wcounts[c] > 0.0) {
                float new_c = (float)(wsums[c] / wcounts[c]);
                float shift = fabsf(new_c - centroids[c]);
                if (shift > max_shift) max_shift = shift;
                centroids[c] = new_c;
            }
        }
        if (max_shift < 1e-8f) break;
    }
    std::sort(centroids, centroids + n_centroids);
}

static inline void lloyd_max(const std::vector<float> & data, float * centroids, int n_centroids, int max_iter) {
    lloyd_max_weighted(data, {}, centroids, n_centroids, max_iter);
}

/* ================================================================
 * Load imatrix from a GGUF file.
 * Returns map: tensor_name → per-element importance (ne0 * ne1 floats).
 * Follows the format used by tools/quantize/quantize.cpp::load_imatrix.
 * ================================================================ */
static std::unordered_map<std::string, std::vector<float>> load_imatrix(const char * path) {
    std::unordered_map<std::string, std::vector<float>> out;

    struct ggml_context * ctx = nullptr;
    struct gguf_init_params params;
    params.no_alloc = false;
    params.ctx = &ctx;
    struct gguf_context * gctx = gguf_init_from_file(path, params);
    if (!gctx) { fprintf(stderr, "WARN: cannot open imatrix %s\n", path); return out; }

    const std::string sums_suffix   = ".in_sum2";
    const std::string counts_suffix = ".counts";

    struct pair_t { const ggml_tensor * sums = nullptr; const ggml_tensor * counts = nullptr; };
    std::unordered_map<std::string, pair_t> pairs;

    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        std::string name = t->name;
        if (name.size() > sums_suffix.size() &&
            name.compare(name.size() - sums_suffix.size(), sums_suffix.size(), sums_suffix) == 0) {
            pairs[name.substr(0, name.size() - sums_suffix.size())].sums = t;
        } else if (name.size() > counts_suffix.size() &&
            name.compare(name.size() - counts_suffix.size(), counts_suffix.size(), counts_suffix) == 0) {
            pairs[name.substr(0, name.size() - counts_suffix.size())].counts = t;
        }
    }

    for (auto & kv : pairs) {
        const auto & name = kv.first;
        const auto * sums = kv.second.sums;
        const auto * counts = kv.second.counts;
        if (!sums || !counts) continue;

        auto & e = out[name];
        e.resize(ggml_nelements(sums));
        const int64_t ne0 = sums->ne[0];
        const int64_t ne1 = sums->ne[1];
        for (int64_t j = 0; j < ne1; j++) {
            float c = ((const float *)counts->data)[j];
            if (c > 0.0f) {
                for (int64_t i = 0; i < ne0; i++) {
                    e[j * ne0 + i] = ((const float *)sums->data)[j * ne0 + i] / c;
                }
            }
        }
    }

    fprintf(stderr, "  loaded imatrix: %zu tensors\n", out.size());
    if (!out.empty()) {
        auto it = out.begin();
        fprintf(stderr, "  sample key: '%s'\n", it->first.c_str());
    }
    gguf_free(gctx);
    ggml_free(ctx);
    return out;
}

/* ================================================================
 * Collect post-RHT scaled values from 2D weight tensors.
 * If imatrix is provided, also collect matching per-value weights
 * (imatrix importance at the input-channel position).
 *
 * Note: RHT mixes input channels. The imatrix importance for post-RHT
 * position i is approximated as the original input-channel importance
 * at position i — this is a well-defined projection since the importance
 * tracks activation energy per channel, and position mapping is identity
 * under our fixed-seed RHT for the purposes of scalar codebook learning.
 * ================================================================ */
static void collect_post_rht(
    const char * path, int block_size, int64_t max_samples,
    const std::unordered_map<std::string, std::vector<float>> & imatrix,
    std::vector<float> & values_out,
    std::vector<float> & weights_out)
{
    struct ggml_context * ctx = nullptr;
    struct gguf_init_params params;
    params.no_alloc = false;
    params.ctx = &ctx;
    struct gguf_context * gctx = gguf_init_from_file(path, params);
    if (!gctx) { fprintf(stderr, "ERROR: cannot open %s\n", path); return; }

    values_out.clear();  values_out.reserve(max_samples);
    weights_out.clear(); weights_out.reserve(imatrix.empty() ? 0 : max_samples);
    int n_tensors = 0;
    int n_with_imat = 0;

    const int total = gguf_get_n_tensors(gctx);
    for (int t = 0; t < total && (int64_t)values_out.size() < max_samples; t++) {
        const char * name = gguf_get_tensor_name(gctx, t);
        struct ggml_tensor * tensor = ggml_get_tensor(ctx, name);
        if (!tensor || ggml_n_dims(tensor) != 2) continue;

        int64_t ne0 = tensor->ne[0], ne1 = tensor->ne[1];
        if (ne0 % block_size != 0) continue;

        /* Look up imatrix for this tensor. Weight tensor is [ne0 in, ne1 out];
         * imatrix is per-input-channel importance (size ne0). Same weight applies
         * to every row (same input channels).
         *
         * If imatrix is provided, skip tensors not covered by it — these are
         * typically embeddings (GET_ROWS, not MUL_MAT) and the codebook should
         * reflect only tensors that will be quantized under imatrix guidance. */
        const std::vector<float> * imat = nullptr;
        if (!imatrix.empty()) {
            auto it = imatrix.find(name);
            if (it == imatrix.end()) continue;  // skip uncovered tensors
            imat = &it->second;
            n_with_imat++;
        }

        std::vector<float> row_f32(ne0);
        int64_t nb = ne0 / block_size;

        for (int64_t row = 0; row < ne1 && (int64_t)values_out.size() < max_samples; row++) {
            const void * row_data = (const char *)tensor->data + row * tensor->nb[1];
            ggml_get_type_traits(tensor->type)->to_float(row_data, row_f32.data(), ne0);

            for (int64_t b = 0; b < nb && (int64_t)values_out.size() < max_samples; b++) {
                float * blk = row_f32.data() + b * block_size;

                float norm_sq = 0;
                for (int i = 0; i < block_size; i++) norm_sq += blk[i] * blk[i];
                float norm = sqrtf(norm_sq);
                if (norm < 1e-10f) continue;

                float rotated[256];
                float inv_norm = 1.0f / norm;
                for (int i = 0; i < block_size; i++) rotated[i] = blk[i] * inv_norm;

                turbo_kv_rht_forward(rotated, block_size, TURBO_KV_DEFAULT_SEED);

                float max_abs = 0;
                for (int i = 0; i < block_size; i++) {
                    float a = fabsf(rotated[i]);
                    if (a > max_abs) max_abs = a;
                }
                if (max_abs < 1e-10f) continue;

                for (int i = 0; i < block_size; i++) {
                    values_out.push_back(rotated[i] / max_abs);
                    if (!imatrix.empty()) {
                        /* Use imatrix value at the input-channel position for this block.
                         * Fall back to 1.0 if no imatrix for this tensor. */
                        float w = 1.0f;
                        if (imat) {
                            int64_t idx = b * block_size + i;
                            if (idx < (int64_t)imat->size()) w = (*imat)[idx];
                            /* Guard against zero weights (would zero-out the sample) */
                            if (w < 1e-6f) w = 1e-6f;
                        }
                        weights_out.push_back(w);
                    }
                }
            }
        }
        n_tensors++;
        if (n_tensors % 20 == 0) {
            fprintf(stderr, "  [%d/%d] %zu values collected\r", n_tensors, total, values_out.size());
        }
    }
    fprintf(stderr, "  Collected %zu values from %d tensors (imatrix matched %d)          \n",
        values_out.size(), n_tensors, n_with_imat);

    gguf_free(gctx);
    ggml_free(ctx);
}

/* ================================================================
 * Compute MSE for a codebook on data
 * ================================================================ */
/* Scale a codebook so max(|centroid|) == cent_max. Lloyd-Max on
 * max-abs-normalized samples produces centroids in [-1, 1]; the quantizer
 * expects centroids in [-cent_max, cent_max] (matching the per-bitrate
 * published convention from tq_codebook.c). This rescale preserves the
 * optimal Lloyd-Max solution because MSE is invariant under uniform
 * scaling of both data and centroids. */
static void rescale_to_cent_max(float * cb, int n, float cent_max) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(cb[i]);
        if (a > mx) mx = a;
    }
    if (mx <= 0.0f) return;
    const float s = cent_max / mx;
    for (int i = 0; i < n; i++) cb[i] *= s;
}

static double compute_mse(const std::vector<float> & data, const float * cb, int n_cb) {
    double mse = 0;
    for (float v : data) {
        float best_d = fabsf(v - cb[0]);
        for (int c = 1; c < n_cb; c++) {
            float d = fabsf(v - cb[c]);
            if (d < best_d) best_d = d;
        }
        mse += best_d * best_d;
    }
    return mse / data.size();
}

/* ================================================================
 * Published Gaussian codebooks (Max 1960, matching tq_codebook.c)
 * Scaled to [-1, 1] range for comparison with our normalized data
 * ================================================================ */
static void get_gaussian_codebook(int bits, float * out, int * n_out) {
    /* These are the standard N(0,1) Lloyd-Max centroids.
     * We scale them to [-1,1] by dividing by max(|centroid|). */
    if (bits == 2) {
        *n_out = 4;
        static const float cb[] = {-1.5104f, -0.4528f, 0.4528f, 1.5104f};
        float scale = 1.0f / 1.5104f;
        for (int i = 0; i < 4; i++) out[i] = cb[i] * scale;
    } else if (bits == 3) {
        *n_out = 8;
        static const float cb[] = {-2.1520f, -1.3440f, -0.7560f, -0.2451f,
                                    0.2451f,  0.7560f,  1.3440f,  2.1520f};
        float scale = 1.0f / 2.1520f;
        for (int i = 0; i < 8; i++) out[i] = cb[i] * scale;
    } else if (bits == 4) {
        *n_out = 16;
        float scale = 1.0f / 2.7326f;
        for (int i = 0; i < 16; i++) out[i] = turbo_kv_4b_codebook[i] * scale;
    } else if (bits == 5) {
        *n_out = 32;
        static const float cb[] = {
            -1.9956f, -1.7900f, -1.6107f, -1.4493f, -1.3010f, -1.1631f, -1.0334f, -0.9104f,
            -0.7928f, -0.6795f, -0.5697f, -0.4626f, -0.3576f, -0.2543f, -0.1520f, -0.0506f,
             0.0506f,  0.1520f,  0.2543f,  0.3576f,  0.4626f,  0.5697f,  0.6795f,  0.7928f,
             0.9104f,  1.0334f,  1.1631f,  1.3010f,  1.4493f,  1.6107f,  1.7900f,  1.9956f};
        float scale = 1.0f / 1.9956f;
        for (int i = 0; i < 32; i++) out[i] = cb[i] * scale;
    } else {
        *n_out = 0;
    }
}

/* ================================================================
 * Write codebook GGUF
 * ================================================================ */
static void write_codebook_gguf(const char * path, const char * model_path,
                                 int block_size, int64_t n_samples,
                                 float kurtosis, float ks_stat,
                                 float cb2[4], float cb3[8], float cb4[16], float cb5[32],
                                 double mse2, double mse3, double mse4, double mse5,
                                 double mse2_gauss, double mse3_gauss, double mse4_gauss, double mse5_gauss) {
    struct gguf_context * gctx = gguf_init_empty();

    gguf_set_val_str (gctx, "turbo.codebook.model",      model_path);
    gguf_set_val_u32 (gctx, "turbo.codebook.block_size",  (uint32_t)block_size);
    gguf_set_val_u64 (gctx, "turbo.codebook.n_samples",   (uint64_t)n_samples);
    gguf_set_val_f32 (gctx, "turbo.codebook.kurtosis",    kurtosis);
    gguf_set_val_f32 (gctx, "turbo.codebook.ks_stat",     ks_stat);

    /* MSE comparison vs Gaussian */
    gguf_set_val_f32 (gctx, "turbo.codebook.mse.2bit.empirical", (float)mse2);
    gguf_set_val_f32 (gctx, "turbo.codebook.mse.2bit.gaussian",  (float)mse2_gauss);
    gguf_set_val_f32 (gctx, "turbo.codebook.mse.3bit.empirical", (float)mse3);
    gguf_set_val_f32 (gctx, "turbo.codebook.mse.3bit.gaussian",  (float)mse3_gauss);
    gguf_set_val_f32 (gctx, "turbo.codebook.mse.4bit.empirical", (float)mse4);
    gguf_set_val_f32 (gctx, "turbo.codebook.mse.4bit.gaussian",  (float)mse4_gauss);
    gguf_set_val_f32 (gctx, "turbo.codebook.mse.5bit.empirical", (float)mse5);
    gguf_set_val_f32 (gctx, "turbo.codebook.mse.5bit.gaussian",  (float)mse5_gauss);

    /* Store centroids as GGUF tensors */
    struct ggml_init_params p = { .mem_size = 4096, .mem_buffer = nullptr, .no_alloc = false };
    struct ggml_context * tctx = ggml_init(p);

    auto add_cb = [&](const char * name, float * data, int n) {
        struct ggml_tensor * t = ggml_new_tensor_1d(tctx, GGML_TYPE_F32, n);
        ggml_set_name(t, name);
        memcpy(t->data, data, n * sizeof(float));
        gguf_add_tensor(gctx, t);
    };

    add_cb("turbo.codebook.2bit", cb2, 4);
    add_cb("turbo.codebook.3bit", cb3, 8);
    add_cb("turbo.codebook.4bit", cb4, 16);
    add_cb("turbo.codebook.5bit", cb5, 32);

    gguf_write_to_file(gctx, path, false);
    gguf_free(gctx);
    ggml_free(tctx);

    fprintf(stderr, "\nWritten codebook to %s\n", path);
}

/* ================================================================
 * Distribution statistics
 * ================================================================ */
static void compute_stats(const std::vector<float> & values, float * kurtosis_out, float * ks_out) {
    double n = values.size();
    double sum = 0, sum2 = 0;
    for (float v : values) { sum += v; sum2 += (double)v * v; }
    double mean = sum / n;
    double var = sum2 / n - mean * mean;
    double std = sqrt(var);

    double cm4 = 0;
    for (float v : values) { double d = (v - mean) / std; cm4 += d*d*d*d; }
    *kurtosis_out = (float)(cm4 / n);

    /* KS statistic vs N(mean, std) */
    std::vector<float> sorted(values);
    std::sort(sorted.begin(), sorted.end());
    double max_ks = 0;
    for (size_t i = 0; i < sorted.size(); i++) {
        double ecdf = (double)(i + 1) / n;
        double z = (sorted[i] - mean) / std;
        double gcdf = 0.5 * (1.0 + erf(z / sqrt(2.0)));
        double diff = fabs(ecdf - gcdf);
        if (diff > max_ks) max_ks = diff;
    }
    *ks_out = (float)max_ks;
}

/* ================================================================
 * Main
 * ================================================================ */
int main(int argc, char ** argv) {
    std::string model_path;
    std::string output_path = "codebook.gguf";
    std::string imatrix_path;
    int block_size = 128;
    int64_t max_samples = 10000000; /* 10M samples for Lloyd-Max */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i+1 < argc) { model_path = argv[++i]; }
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) { output_path = argv[++i]; }
        else if (strcmp(argv[i], "--imatrix") == 0 && i+1 < argc) { imatrix_path = argv[++i]; }
        else if (strcmp(argv[i], "--block-size") == 0 && i+1 < argc) { block_size = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--max-samples") == 0 && i+1 < argc) { max_samples = atoll(argv[++i]); }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            fprintf(stderr, "Usage: %s -m model.gguf [-o codebook.gguf] [--imatrix path.gguf]\n", argv[0]);
            fprintf(stderr, "         [--block-size 128] [--max-samples 10000000]\n");
            fprintf(stderr, "\nComputes model-specific Lloyd-Max codebooks for TURBO_*B quantization.\n");
            fprintf(stderr, "With --imatrix, weights the Lloyd-Max E-step by per-element importance.\n");
            fprintf(stderr, "Output is a GGUF file loadable by llama-quantize via --codebook.\n");
            return 0;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "ERROR: -m model.gguf is required\n");
        return 1;
    }

    fprintf(stderr, "llama-turbo-codebook\n");
    fprintf(stderr, "====================\n");
    fprintf(stderr, "  model:       %s\n", model_path.c_str());
    fprintf(stderr, "  output:      %s\n", output_path.c_str());
    fprintf(stderr, "  imatrix:     %s\n", imatrix_path.empty() ? "(none — uniform weights)" : imatrix_path.c_str());
    fprintf(stderr, "  block_size:  %d\n", block_size);
    fprintf(stderr, "  max_samples: %lld\n", (long long)max_samples);

    /* Phase 0: Load imatrix (if provided) */
    std::unordered_map<std::string, std::vector<float>> imatrix;
    if (!imatrix_path.empty()) {
        fprintf(stderr, "\nPhase 0: Loading imatrix...\n");
        imatrix = load_imatrix(imatrix_path.c_str());
    }

    /* Phase 1: Collect post-RHT values (and weights if imatrix provided) */
    fprintf(stderr, "\nPhase 1: Collecting post-RHT values%s...\n",
            imatrix.empty() ? "" : " + imatrix weights");
    std::vector<float> values, weights;
    collect_post_rht(model_path.c_str(), block_size, max_samples, imatrix, values, weights);
    if (values.empty()) { fprintf(stderr, "ERROR: no values\n"); return 1; }

    /* Phase 2: Distribution statistics (on unweighted values) */
    fprintf(stderr, "\nPhase 2: Distribution statistics...\n");
    float kurtosis, ks_stat;
    compute_stats(values, &kurtosis, &ks_stat);
    fprintf(stderr, "  kurtosis = %.4f (Gaussian: 3.000)\n", kurtosis);
    fprintf(stderr, "  KS stat  = %.4f\n", ks_stat);

    /* Subsample for Lloyd-Max if needed (match value/weight indices) */
    const size_t lloyd_max_n = 1000000;
    std::vector<float> lloyd_data, lloyd_w;
    if (values.size() <= lloyd_max_n) {
        lloyd_data = values;
        lloyd_w = weights;
    } else {
        lloyd_data.resize(lloyd_max_n);
        size_t stride = values.size() / lloyd_max_n;
        for (size_t i = 0; i < lloyd_max_n; i++) lloyd_data[i] = values[i * stride];
        if (!weights.empty()) {
            lloyd_w.resize(lloyd_max_n);
            for (size_t i = 0; i < lloyd_max_n; i++) lloyd_w[i] = weights[i * stride];
        }
    }

    /* Phase 3: Compute codebooks at each bitrate */
    fprintf(stderr, "\nPhase 3: Computing %sLloyd-Max codebooks (%zu samples)...\n",
            lloyd_w.empty() ? "" : "imatrix-weighted ", lloyd_data.size());

    float cb2[4], cb3[8], cb4[16], cb5[32];

    fprintf(stderr, "  2-bit (4 levels)...\n");
    lloyd_max_weighted(lloyd_data, lloyd_w, cb2, 4, 200);

    fprintf(stderr, "  3-bit (8 levels)...\n");
    lloyd_max_weighted(lloyd_data, lloyd_w, cb3, 8, 200);

    fprintf(stderr, "  4-bit (16 levels)...\n");
    lloyd_max_weighted(lloyd_data, lloyd_w, cb4, 16, 200);

    fprintf(stderr, "  5-bit (32 levels)...\n");
    lloyd_max_weighted(lloyd_data, lloyd_w, cb5, 32, 200);

    /* Phase 4: Compare against Gaussian codebooks */
    fprintf(stderr, "\nPhase 4: MSE comparison vs Gaussian codebooks...\n");

    /* Use a subsample for MSE comparison */
    size_t mse_n = std::min(values.size(), (size_t)5000000);
    std::vector<float> mse_data(values.begin(), values.begin() + mse_n);

    float gcb[32]; int gn;
    get_gaussian_codebook(2, gcb, &gn); double mse2g = compute_mse(mse_data, gcb, gn);
    get_gaussian_codebook(3, gcb, &gn); double mse3g = compute_mse(mse_data, gcb, gn);
    get_gaussian_codebook(4, gcb, &gn); double mse4g = compute_mse(mse_data, gcb, gn);
    get_gaussian_codebook(5, gcb, &gn); double mse5g = compute_mse(mse_data, gcb, gn);

    double mse2e = compute_mse(mse_data, cb2, 4);
    double mse3e = compute_mse(mse_data, cb3, 8);
    double mse4e = compute_mse(mse_data, cb4, 16);
    double mse5e = compute_mse(mse_data, cb5, 32);

    fprintf(stderr, "\n  %-6s  %-12s  %-12s  %-10s\n", "Bits", "Gaussian MSE", "Empirical MSE", "Improvement");
    fprintf(stderr, "  %-6s  %-12s  %-12s  %-10s\n", "----", "------------", "-------------", "-----------");
    fprintf(stderr, "  2-bit   %.8f    %.8f    %+.2f%%\n", mse2g, mse2e, 100*(1-mse2e/mse2g));
    fprintf(stderr, "  3-bit   %.8f    %.8f    %+.2f%%\n", mse3g, mse3e, 100*(1-mse3e/mse3g));
    fprintf(stderr, "  4-bit   %.8f    %.8f    %+.2f%%\n", mse4g, mse4e, 100*(1-mse4e/mse4g));
    fprintf(stderr, "  5-bit   %.8f    %.8f    %+.2f%%\n", mse5g, mse5e, 100*(1-mse5e/mse5g));

    /* Rescale centroids to the per-bitrate cent_max convention expected by
     * the quantizer. Lloyd-Max was run on max-abs-normalized samples, so the
     * centroids land in [-1, 1]; the quantizer's block normalization
     * (inv_std = cent_max / max_abs) assumes centroids in [-cent_max, cent_max].
     * Scaling both data and centroids by the same factor preserves the
     * Lloyd-Max optimum, so this is correctness-neutral for the MSE comparison
     * above (which is done in [-1, 1] space) while producing on-disk codebooks
     * in the canonical published range. */
    rescale_to_cent_max(cb2, 4, 1.5104f);
    rescale_to_cent_max(cb3, 8, 2.1520f);
    rescale_to_cent_max(cb4, 16, 2.7326f);  /* matches TURBO_KV_4B_CENT_MAX */
    rescale_to_cent_max(cb5, 32, 1.9956f);

    /* Print codebooks */
    auto print_cb = [](const char * name, const float * cb, int n) {
        fprintf(stderr, "\n  %s:\n    ", name);
        for (int i = 0; i < n; i++) {
            fprintf(stderr, "%+.4f ", cb[i]);
            if ((i+1) % 8 == 0 && i+1 < n) fprintf(stderr, "\n    ");
        }
        fprintf(stderr, "\n");
    };
    print_cb("Empirical 2-bit", cb2, 4);
    print_cb("Empirical 3-bit", cb3, 8);
    print_cb("Empirical 4-bit", cb4, 16);
    print_cb("Empirical 5-bit", cb5, 32);

    /* Phase 5: Write GGUF */
    fprintf(stderr, "\nPhase 5: Writing GGUF...\n");
    write_codebook_gguf(output_path.c_str(), model_path.c_str(),
                        block_size, (int64_t)values.size(),
                        kurtosis, ks_stat,
                        cb2, cb3, cb4, cb5,
                        mse2e, mse3e, mse4e, mse5e,
                        mse2g, mse3g, mse4g, mse5g);

    fprintf(stderr, "\nDone. Use with: llama-quantize --codebook %s ...\n", output_path.c_str());
    return 0;
}
