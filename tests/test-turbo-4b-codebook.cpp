/*
 * test-turbo-4b-codebook.cpp — Empirical post-RHT distribution analysis
 *
 * Task 2: Apply RHT to real weight blocks from a GGUF model, histogram the
 * rotated values, compare against Gaussian, compute Lloyd-Max centroids for
 * the empirical distribution, and measure MSE improvement vs Gaussian codebook.
 *
 * Usage: test-turbo-4b-codebook <model.gguf>
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
#include <numeric>

/* ================================================================
 * Collect post-RHT values from all 2D weight tensors in the model
 * ================================================================ */
static std::vector<float> collect_post_rht_values(const char * model_path, int block_size) {
    struct ggml_context * ctx = nullptr;
    struct gguf_init_params params;
    params.no_alloc = false;
    params.ctx = &ctx;
    struct gguf_context * gctx = gguf_init_from_file(model_path, params);
    if (!gctx) {
        fprintf(stderr, "ERROR: cannot open %s\n", model_path);
        return {};
    }
    std::vector<float> all_rotated;
    int tensors_processed = 0;

    const int n_tensors = gguf_get_n_tensors(gctx);
    for (int t = 0; t < n_tensors; t++) {
        const char * name = gguf_get_tensor_name(gctx, t);
        struct ggml_tensor * tensor = ggml_get_tensor(ctx, name);
        if (!tensor) continue;

        /* Only 2D weight tensors (skip embeddings, 1D biases, etc.) */
        if (ggml_n_dims(tensor) != 2) continue;
        const int64_t ne0 = tensor->ne[0];
        const int64_t ne1 = tensor->ne[1];
        if (ne0 % block_size != 0) continue;

        /* Get f32 data */
        std::vector<float> row_f32(ne0);
        const int64_t nb = ne0 / block_size;

        for (int64_t row = 0; row < ne1; row++) {
            /* Dequant row to f32 */
            const void * row_data = (const char *)tensor->data + row * tensor->nb[1];
            ggml_get_type_traits(tensor->type)->to_float(row_data, row_f32.data(), ne0);

            /* Process each block */
            for (int64_t b = 0; b < nb; b++) {
                float * block_data = row_f32.data() + b * block_size;

                /* L2 norm */
                float norm_sq = 0;
                for (int i = 0; i < block_size; i++) norm_sq += block_data[i] * block_data[i];
                float norm = sqrtf(norm_sq);
                if (norm < 1e-10f) continue; /* skip near-zero blocks */

                /* Normalize */
                float rotated[256];
                float inv_norm = 1.0f / norm;
                for (int i = 0; i < block_size; i++) rotated[i] = block_data[i] * inv_norm;

                /* Forward RHT */
                turbo_kv_rht_forward(rotated, block_size, TURBO_KV_DEFAULT_SEED);

                /* Collect scaled values: x * inv_std where inv_std = CENT_MAX / max_abs */
                float max_abs = 0;
                for (int i = 0; i < block_size; i++) {
                    float a = fabsf(rotated[i]);
                    if (a > max_abs) max_abs = a;
                }
                if (max_abs < 1e-10f) continue;
                float inv_std = 2.7326f / max_abs;

                for (int i = 0; i < block_size; i++) {
                    all_rotated.push_back(rotated[i] * inv_std);
                }
            }
        }
        tensors_processed++;
        fprintf(stderr, "  [%d/%d] %s: %lldx%lld (%lld blocks)\n",
            tensors_processed, n_tensors, name,
            (long long)ne0, (long long)ne1, (long long)(ne0 * ne1 / block_size));
    }

    gguf_free(gctx);
    ggml_free(ctx);
    fprintf(stderr, "  Total: %zu post-RHT scaled values from %d tensors\n",
        all_rotated.size(), tensors_processed);
    return all_rotated;
}

/* ================================================================
 * Histogram and distribution statistics
 * ================================================================ */
static void print_histogram(const std::vector<float> & values, int n_bins) {
    if (values.empty()) return;

    float vmin = *std::min_element(values.begin(), values.end());
    float vmax = *std::max_element(values.begin(), values.end());
    float range = vmax - vmin;
    if (range < 1e-10f) return;

    std::vector<int> bins(n_bins, 0);
    for (float v : values) {
        int b = (int)((v - vmin) / range * (n_bins - 1));
        if (b < 0) b = 0;
        if (b >= n_bins) b = n_bins - 1;
        bins[b]++;
    }

    int max_count = *std::max_element(bins.begin(), bins.end());
    fprintf(stderr, "\n  Histogram (range [%.3f, %.3f], %d bins):\n", vmin, vmax, n_bins);
    for (int b = 0; b < n_bins; b++) {
        float lo = vmin + (float)b / n_bins * range;
        float hi = vmin + (float)(b + 1) / n_bins * range;
        int bar_len = (int)(50.0f * bins[b] / max_count);
        fprintf(stderr, "  [%+.2f,%+.2f) %6d |", lo, hi, bins[b]);
        for (int i = 0; i < bar_len; i++) fputc('#', stderr);
        fputc('\n', stderr);
    }
}

static void compute_distribution_stats(const std::vector<float> & values) {
    if (values.empty()) return;
    double n = values.size();

    /* Mean, variance, skewness, kurtosis */
    double sum = 0, sum2 = 0, sum3 = 0, sum4 = 0;
    for (float v : values) {
        double d = v;
        sum  += d;
        sum2 += d * d;
        sum3 += d * d * d;
        sum4 += d * d * d * d;
    }
    double mean = sum / n;
    double var  = sum2 / n - mean * mean;
    double std  = sqrt(var);

    /* Central moments */
    double cm3 = 0, cm4 = 0;
    for (float v : values) {
        double d = (v - mean) / std;
        cm3 += d * d * d;
        cm4 += d * d * d * d;
    }
    double skewness = cm3 / n;
    double kurtosis = cm4 / n;

    fprintf(stderr, "\n  Distribution statistics:\n");
    fprintf(stderr, "    N          = %zu\n", values.size());
    fprintf(stderr, "    mean       = %.6f  (Gaussian: 0.000)\n", mean);
    fprintf(stderr, "    std        = %.6f  (Gaussian: ~1.0 after scaling)\n", std);
    fprintf(stderr, "    skewness   = %.6f  (Gaussian: 0.000)\n", skewness);
    fprintf(stderr, "    kurtosis   = %.6f  (Gaussian: 3.000)\n", kurtosis);
    fprintf(stderr, "    excess_k   = %.6f  (sub-Gaussian < 0, super-Gaussian > 0)\n", kurtosis - 3.0);

    /* Kolmogorov-Smirnov test vs N(mean, std) */
    std::vector<float> sorted(values);
    std::sort(sorted.begin(), sorted.end());
    double max_ks = 0;
    for (size_t i = 0; i < sorted.size(); i++) {
        double empirical_cdf = (double)(i + 1) / n;
        double z = (sorted[i] - mean) / std;
        double gaussian_cdf = 0.5 * (1.0 + erf(z / sqrt(2.0)));
        double diff = fabs(empirical_cdf - gaussian_cdf);
        if (diff > max_ks) max_ks = diff;
    }
    fprintf(stderr, "    KS stat    = %.6f  (PolarQuant claims < 0.01 at d=128)\n", max_ks);
}

/* ================================================================
 * Lloyd-Max algorithm: iterative optimal quantizer for arbitrary distribution
 * ================================================================ */
static void lloyd_max(const std::vector<float> & values, float * centroids, int n_centroids, int max_iter) {
    /* Initialize centroids uniformly in the data range */
    float vmin = *std::min_element(values.begin(), values.end());
    float vmax = *std::max_element(values.begin(), values.end());
    for (int c = 0; c < n_centroids; c++) {
        centroids[c] = vmin + (vmax - vmin) * (c + 0.5f) / n_centroids;
    }

    std::vector<double> sums(n_centroids);
    std::vector<int> counts(n_centroids);

    for (int iter = 0; iter < max_iter; iter++) {
        /* Assign each value to nearest centroid */
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);

        for (float v : values) {
            int best = 0;
            float best_dist = fabsf(v - centroids[0]);
            for (int c = 1; c < n_centroids; c++) {
                float d = fabsf(v - centroids[c]);
                if (d < best_dist) { best_dist = d; best = c; }
            }
            sums[best] += v;
            counts[best]++;
        }

        /* Update centroids to cluster means */
        float max_shift = 0;
        for (int c = 0; c < n_centroids; c++) {
            if (counts[c] > 0) {
                float new_c = (float)(sums[c] / counts[c]);
                float shift = fabsf(new_c - centroids[c]);
                if (shift > max_shift) max_shift = shift;
                centroids[c] = new_c;
            }
        }
        if (max_shift < 1e-8f) break;
    }

    /* Sort centroids */
    std::sort(centroids, centroids + n_centroids);
}

/* ================================================================
 * MSE comparison: Gaussian codebook vs empirical codebook
 * ================================================================ */
static void compare_codebooks(const std::vector<float> & values,
                               const float * gaussian_cb, const float * empirical_cb, int n_centroids) {
    double mse_gauss = 0, mse_emp = 0;

    for (float v : values) {
        /* Gaussian codebook */
        float best_g = fabsf(v - gaussian_cb[0]);
        for (int c = 1; c < n_centroids; c++) {
            float d = fabsf(v - gaussian_cb[c]);
            if (d < best_g) best_g = d;
        }
        mse_gauss += best_g * best_g;

        /* Empirical codebook */
        float best_e = fabsf(v - empirical_cb[0]);
        for (int c = 1; c < n_centroids; c++) {
            float d = fabsf(v - empirical_cb[c]);
            if (d < best_e) best_e = d;
        }
        mse_emp += best_e * best_e;
    }

    mse_gauss /= values.size();
    mse_emp   /= values.size();

    double improvement = 100.0 * (1.0 - mse_emp / mse_gauss);

    fprintf(stderr, "\n  Codebook comparison (N=%zu values):\n", values.size());
    fprintf(stderr, "    Gaussian MSE  = %.8f\n", mse_gauss);
    fprintf(stderr, "    Empirical MSE = %.8f\n", mse_emp);
    fprintf(stderr, "    Improvement   = %.2f%%\n", improvement);
    fprintf(stderr, "    Verdict: %s\n",
        improvement > 5.0 ? "USE EMPIRICAL (>5%% improvement)" :
        improvement > 1.0 ? "MARGINAL (1-5%% — consider empirical)" :
                            "KEEP GAUSSIAN (<1%% — not worth the complexity)");
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [block_size=128]\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    int block_size = (argc >= 3) ? atoi(argv[2]) : 128;
    fprintf(stderr, "TURBO_4B Codebook Analysis (block_size=%d)\n", block_size);
    fprintf(stderr, "================================================\n");

    /* Phase 2a: Collect post-RHT values */
    fprintf(stderr, "\nPhase 2a: Collecting post-RHT scaled values from %s...\n", model_path);
    std::vector<float> values = collect_post_rht_values(model_path, block_size);
    if (values.empty()) {
        fprintf(stderr, "ERROR: no values collected\n");
        return 1;
    }

    /* Phase 2a: Distribution statistics + histogram */
    compute_distribution_stats(values);
    print_histogram(values, 30);

    /* Phase 2b: Compute Lloyd-Max centroids for empirical distribution.
     * Subsample to 1M values for tractable Lloyd-Max iteration. */
    const size_t max_lloyd_samples = 1000000;
    std::vector<float> lloyd_data;
    if (values.size() <= max_lloyd_samples) {
        lloyd_data = values;
    } else {
        lloyd_data.resize(max_lloyd_samples);
        /* Deterministic stride-based subsampling (not random — reproducible) */
        size_t stride = values.size() / max_lloyd_samples;
        for (size_t i = 0; i < max_lloyd_samples; i++) {
            lloyd_data[i] = values[i * stride];
        }
    }
    fprintf(stderr, "\nPhase 2b: Running Lloyd-Max (16 centroids, 200 iterations, %zu samples)...\n",
        lloyd_data.size());
    float empirical_cb[16];
    lloyd_max(lloyd_data, empirical_cb, 16, 200);

    fprintf(stderr, "\n  Gaussian codebook:\n    ");
    for (int c = 0; c < 16; c++) fprintf(stderr, "%+.4f ", turbo_kv_4b_codebook[c]);
    fprintf(stderr, "\n  Empirical codebook:\n    ");
    for (int c = 0; c < 16; c++) fprintf(stderr, "%+.4f ", empirical_cb[c]);
    fprintf(stderr, "\n  Delta (empirical - Gaussian):\n    ");
    for (int c = 0; c < 16; c++) fprintf(stderr, "%+.4f ", empirical_cb[c] - turbo_kv_4b_codebook[c]);
    fprintf(stderr, "\n");

    /* Phase 2c: MSE comparison (use subsample for speed, full data for accuracy) */
    const size_t max_mse_samples = 10000000;
    std::vector<float> mse_data;
    if (values.size() <= max_mse_samples) {
        mse_data = values;
    } else {
        mse_data.resize(max_mse_samples);
        size_t stride = values.size() / max_mse_samples;
        for (size_t i = 0; i < max_mse_samples; i++) {
            mse_data[i] = values[i * stride];
        }
    }
    compare_codebooks(mse_data, turbo_kv_4b_codebook, empirical_cb, 16);

    return 0;
}
