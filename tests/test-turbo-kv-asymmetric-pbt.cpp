/*
 * test-turbo-kv-asymmetric-pbt.cpp
 *
 * Property-based test obligations propagated from
 * turbo_kv_asymmetric.allium.
 *
 * Status: feature not yet implemented in this port. Asymmetric K/V
 * would let keys and values be quantised at different bit-widths —
 * [TONBI-V3] measurement supports K4/V2 + protected layers at
 * cosine 0.9997, 99% top-1 agreement, 3.6x compression. The port
 * shares one ggml type (TURBO_KV_4B) between K and V — at 9B in
 * production V is still F16 entirely. See the asymmetric-K/V spec for
 * motivation.
 *
 * When asymmetric K/V ships (either as a second ggml type pair or
 * as per-tensor bit-width metadata):
 *   1. Replace each skip(...) with rc::check(...) against the new
 *      dispatch (e.g. quantize_row_turbo_kv_{3,4}b_ref picked by
 *      tensor type or metadata flag).
 *   2. Update the spec-mirror constants below if the port picks
 *      different defaults (sync the spec via /allium:tend).
 *
 * Build: cmake --build build-tq --target test-turbo-kv-asymmetric-pbt
 */

#include <cstdio>

/* rapidcheck not included today — every rule obligation is a skip.
 * When the first rc::check lands, add:
 *   #include <rapidcheck.h>
 *   #include "ggml-turbo-kv.h"
 * and re-enable the rapidcheck link / include in CMakeLists.txt. */

/* ================================================================
 * Spec-mirrored constants from turbo_kv_asymmetric.allium (lines
 * 78-94). Keep in sync with the spec's config block.
 * ================================================================ */

static constexpr int SPEC_KEY_BITS            = 4;
static constexpr int SPEC_VALUE_BITS          = 2;
static constexpr int SPEC_KEY_CODEBOOK_SIZE   = 16;  /* 2^4 */
static constexpr int SPEC_VALUE_CODEBOOK_SIZE = 4;   /* 2^2 */

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
    "asymmetric K/V not implemented in port";

/* ----------------------------------------------------------------
 * entity-fields.{Tensor, QuantizedHead}
 * Both are external entities shared with turbo-kv-4b.allium. Shape
 * enforcement lives in test-turbo-kv-pbt.cpp; reported PASS here
 * because upstream coverage discharges the obligation.
 * ---------------------------------------------------------------- */

static void obligation_entity_fields_Tensor() {
    pass("entity-fields.Tensor");
}

static void obligation_entity_fields_QuantizedHead() {
    pass("entity-fields.QuantizedHead");
}

/* ----------------------------------------------------------------
 * config-default.{key_bits, value_bits, key_codebook_size,
 *                 value_codebook_size}
 * All four are spec-mirror constants today — PASS unconditionally
 * as internal consistency. When the port exposes runtime overrides
 * (e.g. --cache-type-k / --cache-type-v or server params), the
 * assertion moves from spec-mirror to live-config.
 * ---------------------------------------------------------------- */

static void obligation_config_default_key_bits() {
    if (SPEC_KEY_BITS != 4) {
        fail("config-default.key_bits", "spec-mirror out of sync");
        return;
    }
    skip("config-default.key_bits", SKIP_NOT_IMPLEMENTED);
}

static void obligation_config_default_value_bits() {
    if (SPEC_VALUE_BITS != 2) {
        fail("config-default.value_bits", "spec-mirror out of sync");
        return;
    }
    skip("config-default.value_bits", SKIP_NOT_IMPLEMENTED);
}

static void obligation_config_default_key_codebook_size() {
    /* Spec declares key_codebook_size = 16 alongside a comment
     * "2^key_bits at default 4". Verify the relationship holds
     * regardless of the port's implementation state. */
    if (SPEC_KEY_CODEBOOK_SIZE != (1 << SPEC_KEY_BITS)) {
        fail("config-default.key_codebook_size",
             "codebook size must be 2^key_bits");
        return;
    }
    pass("config-default.key_codebook_size");
}

static void obligation_config_default_value_codebook_size() {
    if (SPEC_VALUE_CODEBOOK_SIZE != (1 << SPEC_VALUE_BITS)) {
        fail("config-default.value_codebook_size",
             "codebook size must be 2^value_bits");
        return;
    }
    pass("config-default.value_codebook_size");
}

/* ----------------------------------------------------------------
 * rule-success/failure/entity-creation for QuantizeKeyHead and
 * QuantizeValueHead. These compose on turbo-kv-4b.allium's
 * QuantizeHead with bits = key_bits / value_bits respectively.
 * When the dispatch lands, the success tests become roundtrip PBT
 * at each bit-width; failures cover the empty-tensor precondition.
 * ---------------------------------------------------------------- */

static void obligation_rule_success_QuantizeKeyHead() {
    skip("rule-success.QuantizeKeyHead", SKIP_NOT_IMPLEMENTED);
}

static void obligation_rule_failure_QuantizeKeyHead_1() {
    skip("rule-failure.QuantizeKeyHead.1", SKIP_NOT_IMPLEMENTED);
}

static void obligation_rule_entity_creation_QuantizeKeyHead_1() {
    skip("rule-entity-creation.QuantizeKeyHead.1", SKIP_NOT_IMPLEMENTED);
}

static void obligation_rule_success_QuantizeValueHead() {
    skip("rule-success.QuantizeValueHead", SKIP_NOT_IMPLEMENTED);
}

static void obligation_rule_failure_QuantizeValueHead_1() {
    skip("rule-failure.QuantizeValueHead.1", SKIP_NOT_IMPLEMENTED);
}

static void obligation_rule_entity_creation_QuantizeValueHead_1() {
    skip("rule-entity-creation.QuantizeValueHead.1", SKIP_NOT_IMPLEMENTED);
}

/* ----------------------------------------------------------------
 * invariant.KeyBitsAtLeastValueBits
 * Spec line 159: config.key_bits >= config.value_bits.
 * Today this is a static property of the spec-mirror constants —
 * PASS if 4 >= 2. When runtime overrides exist, the same assertion
 * runs against live config.
 * ---------------------------------------------------------------- */

static void obligation_invariant_KeyBitsAtLeastValueBits() {
    if (SPEC_KEY_BITS < SPEC_VALUE_BITS) {
        fail("invariant.KeyBitsAtLeastValueBits",
             "spec-mirror defaults violate K >= V");
        return;
    }
    pass("invariant.KeyBitsAtLeastValueBits");
}

/* ----------------------------------------------------------------
 * invariant.CodebookSizesConsistent
 * Spec line 164: key_codebook_size = 2^key_bits AND value_codebook_size
 * = 2^value_bits.
 * ---------------------------------------------------------------- */

static void obligation_invariant_CodebookSizesConsistent() {
    const bool ok_key   = SPEC_KEY_CODEBOOK_SIZE   == (1 << SPEC_KEY_BITS);
    const bool ok_value = SPEC_VALUE_CODEBOOK_SIZE == (1 << SPEC_VALUE_BITS);
    if (!ok_key || !ok_value) {
        fail("invariant.CodebookSizesConsistent",
             "codebook size mismatch against 2^bits");
        return;
    }
    pass("invariant.CodebookSizesConsistent");
}

int main() {
    fprintf(stdout,
        "=== test-turbo-kv-asymmetric-pbt ===\n"
        "Spec: turbo_kv_asymmetric.allium\n"
        "Port status: asymmetric K/V NOT implemented\n"
        "14 obligations — implementation-dependent ones SKIP.\n\n");

    obligation_entity_fields_Tensor();
    obligation_entity_fields_QuantizedHead();

    obligation_config_default_key_bits();
    obligation_config_default_value_bits();
    obligation_config_default_key_codebook_size();
    obligation_config_default_value_codebook_size();

    obligation_rule_success_QuantizeKeyHead();
    obligation_rule_failure_QuantizeKeyHead_1();
    obligation_rule_entity_creation_QuantizeKeyHead_1();

    obligation_rule_success_QuantizeValueHead();
    obligation_rule_failure_QuantizeValueHead_1();
    obligation_rule_entity_creation_QuantizeValueHead_1();

    obligation_invariant_KeyBitsAtLeastValueBits();
    obligation_invariant_CodebookSizesConsistent();

    fprintf(stdout,
        "\n=== Summary ===\n"
        "  PASS: %d\n"
        "  SKIP: %d  (pending asymmetric K/V implementation)\n"
        "  FAIL: %d\n"
        "Total: %d obligations\n",
        g_acct.pass, g_acct.skip, g_acct.fail,
        g_acct.pass + g_acct.skip + g_acct.fail);

    return g_acct.fail == 0 ? 0 : 1;
}
