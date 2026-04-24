/*
 * test-turbo-kv-outliers-pbt.cpp
 *
 * Property-based test obligations propagated from
 * turbo_kv_outliers.allium.
 *
 * Status: feature not yet implemented. Outlier handling stores the
 * K=8 largest-magnitude rotated channels verbatim as fp16 alongside
 * their channel indices, OVERWRITING the codebook reconstruction at
 * those channels on dequantise. Addresses the heavy-tail problem in
 * attention activations that a single global codebook fits poorly.
 * [QUANT.CPP] ships this as block_tq_turbo_kv_4bo (96 B / 128 elem,
 * +24 B over 4b) and block_tq_turbo_kv_3bo (80 B / 128 elem,
 * +24 B over 3b). Port has no TURBO_KV_4BO / TURBO_KV_3BO types.
 * See the outlier-handling spec (highest implementation cost of the four
 * sibling improvements — "about the same effort as adding a whole
 * new precision tier").
 *
 * When outlier types ship:
 *   1. Add GGML_TYPE_TURBO_KV_4BO (and 3BO), wire them through the
 *      quantise / dequantise / vec_dot / FA paths.
 *   2. Replace skip(...) with rc::check(...) against the new
 *      dispatch.
 *   3. Sync spec-mirror constants if defaults change.
 *
 * Build: cmake --build build-tq --target test-turbo-kv-outliers-pbt
 */

#include <cstdio>

/* rapidcheck not included today — skip-only stub. */

/* ================================================================
 * Spec-mirrored constants from turbo_kv_outliers.allium (lines
 * 98-113).
 * ================================================================ */

static constexpr int   SPEC_OUTLIER_COUNT                = 8;
static constexpr float SPEC_MIN_OUTLIER_MAGNITUDE_FRAC   = 0.1f;

namespace {
struct Accounting {
    int pass = 0;
    int skip = 0;
    int fail = 0;
};

Accounting g_acct;
}

static void pass(const char * obligation_id) {
    fprintf(stdout, "[PASS] %s\n", obligation_id);
    g_acct.pass++;
}

static void skip(const char * obligation_id, const char * reason) {
    fprintf(stdout, "[SKIP] %s — %s\n", obligation_id, reason);
    g_acct.skip++;
}

static void fail(const char * obligation_id, const char * reason) {
    fprintf(stdout, "[FAIL] %s — %s\n", obligation_id, reason);
    g_acct.fail++;
}

static constexpr const char * SKIP_NOT_IMPLEMENTED =
    "outlier handling not implemented in port";

/* ----------------------------------------------------------------
 * entity-fields.{Tensor, OutlierEntry, QuantizedHeadWithOutliers}
 * Tensor is inherited external; the other two belong to this spec.
 * PASS for Tensor (upstream coverage), SKIP for the outlier-native
 * entities because no corresponding C struct exists in the port.
 * ---------------------------------------------------------------- */

static void obligation_entity_fields_Tensor() {
    pass("entity-fields.Tensor");
}

static void obligation_entity_fields_OutlierEntry() {
    /* OutlierEntry { channel_index: Integer, value: Decimal }.
     * No C struct today; [QUANT.CPP]'s layout would be a uint8
     * index + fp16 value per entry = 3 bytes. SKIP. */
    skip("entity-fields.OutlierEntry", SKIP_NOT_IMPLEMENTED);
}

static void obligation_entity_fields_QuantizedHeadWithOutliers() {
    /* QuantizedHeadWithOutliers has { norm, inv_std, indices,
     * outliers: Set<OutlierEntry> }. No block_turbo_kv_4bo today. */
    skip("entity-fields.QuantizedHeadWithOutliers", SKIP_NOT_IMPLEMENTED);
}

/* ----------------------------------------------------------------
 * config-default.outlier_count
 * Spec line 105 — default 8 (matches [QUANT.CPP]'s
 * TQ_KV_4BO_OUTLIERS #define). Today spec-mirror only.
 * ---------------------------------------------------------------- */

static void obligation_config_default_outlier_count() {
    if (SPEC_OUTLIER_COUNT != 8) {
        fail("config-default.outlier_count", "spec-mirror out of sync");
        return;
    }
    skip("config-default.outlier_count", SKIP_NOT_IMPLEMENTED);
}

/* ----------------------------------------------------------------
 * config-default.min_outlier_magnitude_fraction
 * Spec line 112 — default 0.1. Diagnostic invariant threshold
 * ("outliers typically capture >40% of total magnitude" per spec
 * comment; 0.1 is a loose safety bound). SKIP today.
 * ---------------------------------------------------------------- */

static void obligation_config_default_min_outlier_magnitude_fraction() {
    if (SPEC_MIN_OUTLIER_MAGNITUDE_FRAC != 0.1f) {
        fail("config-default.min_outlier_magnitude_fraction",
             "spec-mirror out of sync");
        return;
    }
    skip("config-default.min_outlier_magnitude_fraction", SKIP_NOT_IMPLEMENTED);
}

/* ----------------------------------------------------------------
 * rule-success/failure/entity-creation for QuantizeHeadWithOutliers.
 * Spec lines 141-161. Two requires (tensor.count > 0, L2_norm > 0)
 * → two rule-failure obligations.
 * ---------------------------------------------------------------- */

static void obligation_rule_success_QuantizeHeadWithOutliers() {
    skip("rule-success.QuantizeHeadWithOutliers", SKIP_NOT_IMPLEMENTED);
}

static void obligation_rule_failure_QuantizeHeadWithOutliers_1() {
    skip("rule-failure.QuantizeHeadWithOutliers.1", SKIP_NOT_IMPLEMENTED);
}

static void obligation_rule_failure_QuantizeHeadWithOutliers_2() {
    skip("rule-failure.QuantizeHeadWithOutliers.2", SKIP_NOT_IMPLEMENTED);
}

static void obligation_rule_entity_creation_QuantizeHeadWithOutliers_1() {
    skip("rule-entity-creation.QuantizeHeadWithOutliers.1", SKIP_NOT_IMPLEMENTED);
}

/* ----------------------------------------------------------------
 * rule-success/failure/entity-creation for DequantizeHeadWithOutliers.
 * Spec lines 169-184. Two requires (block.inv_std > 0,
 * block.indices.count > 0) → two rule-failure obligations.
 * ---------------------------------------------------------------- */

static void obligation_rule_success_DequantizeHeadWithOutliers() {
    skip("rule-success.DequantizeHeadWithOutliers", SKIP_NOT_IMPLEMENTED);
}

static void obligation_rule_failure_DequantizeHeadWithOutliers_1() {
    skip("rule-failure.DequantizeHeadWithOutliers.1", SKIP_NOT_IMPLEMENTED);
}

static void obligation_rule_failure_DequantizeHeadWithOutliers_2() {
    skip("rule-failure.DequantizeHeadWithOutliers.2", SKIP_NOT_IMPLEMENTED);
}

static void obligation_rule_entity_creation_DequantizeHeadWithOutliers_1() {
    skip("rule-entity-creation.DequantizeHeadWithOutliers.1", SKIP_NOT_IMPLEMENTED);
}

/* ----------------------------------------------------------------
 * invariant.OutlierIndicesInBounds
 *   Spec lines 191-196: every outlier.channel_index in [0,
 *   block.indices.count). Needs a live QuantizedHeadWithOutliers
 *   instance to check.
 *
 * invariant.OutlierCountConsistent
 *   Spec lines 199-202: block.outliers.count == config.outlier_count.
 *
 * invariant.OutliersUnique
 *   Spec lines 206-211: no two outlier entries share a channel_index.
 * ---------------------------------------------------------------- */

static void obligation_invariant_OutlierIndicesInBounds() {
    skip("invariant.OutlierIndicesInBounds", SKIP_NOT_IMPLEMENTED);
}

static void obligation_invariant_OutlierCountConsistent() {
    skip("invariant.OutlierCountConsistent", SKIP_NOT_IMPLEMENTED);
}

static void obligation_invariant_OutliersUnique() {
    skip("invariant.OutliersUnique", SKIP_NOT_IMPLEMENTED);
}

int main() {
    fprintf(stdout,
        "=== test-turbo-kv-outliers-pbt ===\n"
        "Spec: turbo_kv_outliers.allium\n"
        "Port status: outlier handling NOT implemented\n"
        "16 obligations — implementation-dependent ones SKIP.\n\n");

    obligation_entity_fields_Tensor();
    obligation_entity_fields_OutlierEntry();
    obligation_entity_fields_QuantizedHeadWithOutliers();

    obligation_config_default_outlier_count();
    obligation_config_default_min_outlier_magnitude_fraction();

    obligation_rule_success_QuantizeHeadWithOutliers();
    obligation_rule_failure_QuantizeHeadWithOutliers_1();
    obligation_rule_failure_QuantizeHeadWithOutliers_2();
    obligation_rule_entity_creation_QuantizeHeadWithOutliers_1();

    obligation_rule_success_DequantizeHeadWithOutliers();
    obligation_rule_failure_DequantizeHeadWithOutliers_1();
    obligation_rule_failure_DequantizeHeadWithOutliers_2();
    obligation_rule_entity_creation_DequantizeHeadWithOutliers_1();

    obligation_invariant_OutlierIndicesInBounds();
    obligation_invariant_OutlierCountConsistent();
    obligation_invariant_OutliersUnique();

    fprintf(stdout,
        "\n=== Summary ===\n"
        "  PASS: %d\n"
        "  SKIP: %d  (pending outlier handling implementation)\n"
        "  FAIL: %d\n"
        "Total: %d obligations\n",
        g_acct.pass, g_acct.skip, g_acct.fail,
        g_acct.pass + g_acct.skip + g_acct.fail);

    return g_acct.fail == 0 ? 0 : 1;
}
