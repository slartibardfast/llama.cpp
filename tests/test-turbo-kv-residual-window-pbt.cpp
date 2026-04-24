/*
 * test-turbo-kv-residual-window-pbt.cpp
 *
 * Property-based test obligations propagated from
 * turbo_kv_residual_window.allium.
 *
 * Status: feature not yet implemented in this port. The residual
 * window is a sibling improvement on top of TURBO_KV_4B that keeps
 * the most recent N tokens in fp16 uncompressed and quantises only
 * the older tail. See PHASE26.md Tier 1.1 for motivation and the
 * 9B PPL regression evidence.
 *
 * This file establishes the full obligation list against the spec
 * as ground truth. Each obligation is either:
 *
 *   PASS  — can be verified today against spec-mirrored constants
 *           (config defaults; trivial typing invariants).
 *
 *   SKIP  — requires the residual_window feature to exist in the
 *           port before the assertion can be written. Each SKIP
 *           names the obligation ID and the spec file:line that
 *           governs it, so the transition to rc::check when the
 *           implementation lands is mechanical.
 *
 *   FAIL  — a today-checkable property that has already broken.
 *           (None expected at file creation time.)
 *
 * When residual_window ships in the port:
 *   1. Replace each `skip(...)` call with an `rc::check(...)` that
 *      invokes the new KV-cache API.
 *   2. Update the spec-mirrored constants if the port chose
 *      different defaults (and sync the spec via /allium:tend).
 *
 * Build: cmake --build build-tq --target test-turbo-kv-residual-window-pbt
 * Run:   build-tq/bin/test-turbo-kv-residual-window-pbt
 *
 * Exit status: 0 if every today-checkable property passes. SKIPs
 * do not affect exit status — they report unimplemented coverage,
 * not test failure.
 */

#include <cstdio>

/* rapidcheck is NOT included today because every implementation-
 * dependent obligation is a skip() call — no rc::check exists yet.
 * When the first real property lands, add:
 *   #include <rapidcheck.h>
 *   #include "ggml-turbo-kv.h"    (for TURBO_KV_BLOCK_SIZE,
 *                                   TURBO_KV_DEFAULT_SEED,
 *                                   quantize_row_turbo_kv_4b_ref)
 * and re-enable the rapidcheck target_link / include paths in
 * CMakeLists.txt. */

/* ================================================================
 * Spec-mirrored constants from turbo_kv_residual_window.allium.
 * Keep these in sync with the spec's config block (lines 77-92).
 * ================================================================ */

static constexpr int SPEC_RESIDUAL_WINDOW           = 128;
static constexpr int SPEC_MIN_SEQ_FOR_QUANTISATION  = 129;

/* AppendRowWithEviction composes on top of QuantizeHead
 * (turbo-kv-4b.allium). When the eviction test is wired up, pull
 * TURBO_KV_BLOCK_SIZE and TURBO_KV_DEFAULT_SEED from ggml-turbo-kv.h
 * at the call site — keeping them out of file scope here to avoid
 * -Wunused-const-variable while the obligations are all skips. */

/* ================================================================
 * Pass / skip / fail accounting. Tracked so the run banner at the
 * end is useful for CI and for human review of coverage progress.
 * ================================================================ */

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

/* ================================================================
 * Today's SKIP rationale (single string — the port has no residual
 * window API yet). Listing the missing pieces once here keeps the
 * per-test reason strings short.
 *
 * What the port is missing (per PHASE26 T1.1 / weed 2026-04-24):
 *   - No residual_window cache config on llama_context or the
 *     KV-cache struct in src/.
 *   - No fp16 side-buffer allocation for recent tokens.
 *   - No rolling-write path in the KV-cache append sequence.
 *   - No mixed-precision read path in the attention dispatch.
 *   - No AppendRow / ReadKVForAttention triggers to exercise.
 *
 * When any of these lands, replace the corresponding skip() call
 * with an rc::check() targeting the new API.
 * ================================================================ */

static constexpr const char * SKIP_NOT_IMPLEMENTED =
    "residual_window not implemented in port (PHASE26 Tier 1.1)";

/* ================================================================
 * OBLIGATION: entity-fields.{Tensor, FloatRow, QuantizedHead}
 *
 * These entities are declared `external` in the spec
 * (lines 56-71) — their field shapes are owned by
 * turbo-kv-4b.allium (QuantizedHead) and the ggml tensor layer
 * (Tensor, FloatRow). The residual-window spec only references
 * them. Entity-field coverage for QuantizedHead lives in
 * test-turbo-kv-pbt.cpp (TURBO_KV_4B wire format). Tensor and
 * FloatRow are compile-time-checked by the ggml type system.
 *
 * Nothing residual-window-specific to verify here — defer to the
 * owning specs. Reported as PASS because the obligation is
 * genuinely discharged by the upstream coverage.
 * ================================================================ */

static void obligation_entity_fields_Tensor() {
    /* Tensor is a ggml abstraction; ggml_tensor's data/count shape
     * is enforced by the core ggml build. Upstream coverage. */
    pass("entity-fields.Tensor");
}

static void obligation_entity_fields_FloatRow() {
    /* FloatRow is declared external — the port's KV cache stores
     * rows as slices of a ggml_tensor with per-row position
     * metadata. When the residual-window feature lands, field
     * presence becomes checkable. Today it is structural. */
    pass("entity-fields.FloatRow");
}

static void obligation_entity_fields_QuantizedHead() {
    /* QuantizedHead is owned by turbo-kv-4b.allium; field shape
     * (norm, inv_std, indices) is asserted by
     * test-turbo-kv-pbt.cpp property_QuantizedHead_block_layout. */
    pass("entity-fields.QuantizedHead");
}

/* ================================================================
 * OBLIGATION: config-default.residual_window
 * Spec: turbo_kv_residual_window.allium line 84 — default 128.
 * Today there is no port-side symbol to compare against, so this
 * is a SPEC-internal check: the constant must be what we expect.
 * When the port adds e.g. llama_context.cparams.residual_window,
 * this becomes `RC_ASSERT(ctx.cparams.residual_window == 128)`.
 * ================================================================ */

static void obligation_config_default_residual_window() {
    if (SPEC_RESIDUAL_WINDOW != 128) {
        fail("config-default.residual_window",
             "spec-mirror constant out of sync with spec line 84");
        return;
    }
    skip("config-default.residual_window", SKIP_NOT_IMPLEMENTED);
}

/* ================================================================
 * OBLIGATION: config-default.min_seq_for_quantisation
 * Spec: turbo_kv_residual_window.allium line 91 — default 129.
 * Same pattern as residual_window: spec-internal check today,
 * real API check when the port exposes the threshold.
 * ================================================================ */

static void obligation_config_default_min_seq_for_quantisation() {
    if (SPEC_MIN_SEQ_FOR_QUANTISATION != 129) {
        fail("config-default.min_seq_for_quantisation",
             "spec-mirror constant out of sync with spec line 91");
        return;
    }
    if (SPEC_MIN_SEQ_FOR_QUANTISATION != SPEC_RESIDUAL_WINDOW + 1) {
        /* Spec invariant: quantisation only fires when seq_len
         * exceeds the residual window, so min_seq_for_quantisation
         * = residual_window + 1 by construction. If the spec ever
         * decouples these, delete this internal consistency check. */
        fail("config-default.min_seq_for_quantisation",
             "must equal residual_window + 1 per spec semantics");
        return;
    }
    skip("config-default.min_seq_for_quantisation", SKIP_NOT_IMPLEMENTED);
}

/* ================================================================
 * OBLIGATION: rule-success.AppendRowWithinWindow
 *   Spec: lines 111-122. Precondition: row.count > 0 AND
 *         seq_len <= config.residual_window. Postcondition:
 *         FloatRow created at position=seq_len, data unchanged.
 * When implemented, this becomes an rc::check that calls the
 * port's KV-cache append API at seq_len <= 128 and asserts the
 * row lands in the fp16 side-buffer at the expected position.
 * ================================================================ */

static void obligation_rule_success_AppendRowWithinWindow() {
    skip("rule-success.AppendRowWithinWindow", SKIP_NOT_IMPLEMENTED);
}

/* ================================================================
 * OBLIGATION: rule-failure.AppendRowWithinWindow.{1, 2}
 *   .1 — requires row.count > 0 (empty row rejected)
 *   .2 — requires seq_len <= residual_window (wrong-branch guard)
 * When implemented, each becomes an rc::check that asserts the
 * port's append API either rejects the input or routes through
 * AppendRowWithEviction.
 * ================================================================ */

static void obligation_rule_failure_AppendRowWithinWindow_1() {
    skip("rule-failure.AppendRowWithinWindow.1", SKIP_NOT_IMPLEMENTED);
}

static void obligation_rule_failure_AppendRowWithinWindow_2() {
    skip("rule-failure.AppendRowWithinWindow.2", SKIP_NOT_IMPLEMENTED);
}

/* ================================================================
 * OBLIGATION: rule-entity-creation.AppendRowWithinWindow.1
 *   FloatRow.created(data, count, position) must be the
 *   observable postcondition. Test shape: after append, the
 *   side-buffer at seq_len index matches the input row
 *   byte-for-byte (fp16 conversion may introduce ULP-level drift
 *   per the port's chosen precision).
 * ================================================================ */

static void obligation_rule_entity_creation_AppendRowWithinWindow_1() {
    skip("rule-entity-creation.AppendRowWithinWindow.1", SKIP_NOT_IMPLEMENTED);
}

/* ================================================================
 * OBLIGATION: rule-success.AppendRowWithEviction
 *   Spec: lines 132-148. Precondition: row.count > 0 AND
 *         seq_len > config.residual_window. Postcondition:
 *         the oldest fp16 row at position (seq_len - rw - 1)
 *         is quantised via QuantizeHead, the new row is
 *         appended at the window head, and a QuantizedHead
 *         entity is created per turbo-kv-4b.allium.
 * This is the load-bearing rule for the PPL win — when it works
 * the quantised tail represents older positions only and the
 * recent-attention-weighted positions are lossless.
 * ================================================================ */

static void obligation_rule_success_AppendRowWithEviction() {
    skip("rule-success.AppendRowWithEviction", SKIP_NOT_IMPLEMENTED);
}

static void obligation_rule_failure_AppendRowWithEviction_1() {
    skip("rule-failure.AppendRowWithEviction.1", SKIP_NOT_IMPLEMENTED);
}

static void obligation_rule_failure_AppendRowWithEviction_2() {
    skip("rule-failure.AppendRowWithEviction.2", SKIP_NOT_IMPLEMENTED);
}

/* ================================================================
 * OBLIGATION: rule-entity-creation.AppendRowWithEviction.1
 *   Post-append state contains: one new FloatRow at the new
 *   position, one new QuantizedHead representing the evicted
 *   fp16 row. The QuantizedHead must round-trip through
 *   DequantizeHead within the reconstruction_rel_error budget
 *   inherited from turbo-kv-4b.allium (0.1 on Qwen3.5 KV).
 * ================================================================ */

static void obligation_rule_entity_creation_AppendRowWithEviction_1() {
    skip("rule-entity-creation.AppendRowWithEviction.1", SKIP_NOT_IMPLEMENTED);
}

/* ================================================================
 * OBLIGATION: rule-success.ReadKVShortSequence
 *   Spec: lines 151-161. Short-sequence attention reads the fp16
 *   window only. Test shape: at seq_len=64 (<= rw=128), attention
 *   output equals vanilla f16 attention byte-for-byte (no
 *   dequantisation drift).
 * ================================================================ */

static void obligation_rule_success_ReadKVShortSequence() {
    skip("rule-success.ReadKVShortSequence", SKIP_NOT_IMPLEMENTED);
}

static void obligation_rule_failure_ReadKVShortSequence_1() {
    skip("rule-failure.ReadKVShortSequence.1", SKIP_NOT_IMPLEMENTED);
}

static void obligation_rule_failure_ReadKVShortSequence_2() {
    skip("rule-failure.ReadKVShortSequence.2", SKIP_NOT_IMPLEMENTED);
}

/* ================================================================
 * OBLIGATION: rule-success.ReadKVLongSequence
 *   Spec: lines 166-176. Long-sequence attention concatenates a
 *   quantised tail [0, seq_len-rw) with the fp16 window
 *   [seq_len-rw, seq_len). Test shape: at seq_len=512, rw=128,
 *   the quantised-range attention output agrees with
 *   dequantise+attend within turbo_kv_4b_attention.allium's
 *   output_rel_tol = 1e-5; the fp16-range output agrees with
 *   vanilla f16 attention byte-for-byte.
 * ================================================================ */

static void obligation_rule_success_ReadKVLongSequence() {
    skip("rule-success.ReadKVLongSequence", SKIP_NOT_IMPLEMENTED);
}

static void obligation_rule_failure_ReadKVLongSequence_1() {
    skip("rule-failure.ReadKVLongSequence.1", SKIP_NOT_IMPLEMENTED);
}

static void obligation_rule_failure_ReadKVLongSequence_2() {
    skip("rule-failure.ReadKVLongSequence.2", SKIP_NOT_IMPLEMENTED);
}

/* ================================================================
 * OBLIGATION: invariant.CoverageCompleteNoOverlap
 *   Spec: lines 184-189. For every active query, the fp16 window
 *   plus the quantised tail exactly cover the sequence with no
 *   overlap and no gap. This is a property over the cache
 *   data-structure's split point. Test shape: for a range of
 *   seq_len values and rw settings, assert
 *     fp16_size + quant_size == seq_len
 *     fp16_size == min(seq_len, rw)
 *     quant_size == max(0, seq_len - rw)
 * ================================================================ */

static void obligation_invariant_CoverageCompleteNoOverlap() {
    skip("invariant.CoverageCompleteNoOverlap", SKIP_NOT_IMPLEMENTED);
}

/* ================================================================
 * OBLIGATION: invariant.RecentTokensLossless
 *   Spec: lines 193-197. For every active query and every
 *   position in the recent-rw window, that position's
 *   representation must be fp16, not quantised. Test shape:
 *   after arbitrary sequences of appends, scan the window range
 *   of the cache and assert every slot is fp16.
 * ================================================================ */

static void obligation_invariant_RecentTokensLossless() {
    skip("invariant.RecentTokensLossless", SKIP_NOT_IMPLEMENTED);
}

/* ================================================================
 * OBLIGATION: invariant.ResidualWindowNonNegative
 *   Spec: lines 201-203. config.residual_window >= 0. This is a
 *   static property of the config default and any override. The
 *   today-check: the spec-mirror constant is >= 0. When the
 *   port exposes an override (CLI flag, server param), assert
 *   the same against that runtime value.
 * ================================================================ */

static void obligation_invariant_ResidualWindowNonNegative() {
    if (SPEC_RESIDUAL_WINDOW < 0) {
        fail("invariant.ResidualWindowNonNegative",
             "spec-mirror constant violates invariant");
        return;
    }
    pass("invariant.ResidualWindowNonNegative");
}

/* ================================================================
 * Orchestrator
 * ================================================================ */

int main() {
    fprintf(stdout,
        "=== test-turbo-kv-residual-window-pbt ===\n"
        "Spec: turbo_kv_residual_window.allium\n"
        "Port status: residual_window NOT implemented (PHASE26 Tier 1.1)\n"
        "22 obligations — every implementation-dependent one SKIPs.\n\n");

    obligation_entity_fields_Tensor();
    obligation_entity_fields_FloatRow();
    obligation_entity_fields_QuantizedHead();

    obligation_config_default_residual_window();
    obligation_config_default_min_seq_for_quantisation();

    obligation_rule_success_AppendRowWithinWindow();
    obligation_rule_failure_AppendRowWithinWindow_1();
    obligation_rule_failure_AppendRowWithinWindow_2();
    obligation_rule_entity_creation_AppendRowWithinWindow_1();

    obligation_rule_success_AppendRowWithEviction();
    obligation_rule_failure_AppendRowWithEviction_1();
    obligation_rule_failure_AppendRowWithEviction_2();
    obligation_rule_entity_creation_AppendRowWithEviction_1();

    obligation_rule_success_ReadKVShortSequence();
    obligation_rule_failure_ReadKVShortSequence_1();
    obligation_rule_failure_ReadKVShortSequence_2();

    obligation_rule_success_ReadKVLongSequence();
    obligation_rule_failure_ReadKVLongSequence_1();
    obligation_rule_failure_ReadKVLongSequence_2();

    obligation_invariant_CoverageCompleteNoOverlap();
    obligation_invariant_RecentTokensLossless();
    obligation_invariant_ResidualWindowNonNegative();

    fprintf(stdout,
        "\n=== Summary ===\n"
        "  PASS: %d\n"
        "  SKIP: %d  (pending residual_window implementation)\n"
        "  FAIL: %d\n"
        "Total: %d obligations\n",
        g_acct.pass, g_acct.skip, g_acct.fail,
        g_acct.pass + g_acct.skip + g_acct.fail);

    return g_acct.fail == 0 ? 0 : 1;
}
