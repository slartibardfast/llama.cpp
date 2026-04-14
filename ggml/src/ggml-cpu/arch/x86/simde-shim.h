/*
 * simde-shim.h — Westmere-to-AVX translation gateway.
 *
 * This header is included ONLY by translation units that want to use
 * AVX / AVX2 / F16C / FMA intrinsics on a target CPU that doesn't have
 * those instruction sets natively. SIMDe (SIMD-Everywhere) translates
 * the intrinsics to portable sequences of SSE2/SSE4.1 ops that Westmere
 * can execute.
 *
 * The build system arranges for including TUs to be compiled with
 * -march=native so that SIMDe can pick up available helpers (POPCNT,
 * SSE4.1, etc.) rather than the portable baseline.
 *
 * Do NOT include this from a header that's pulled in widely — the
 * NATIVE_ALIASES defines pollute the global intrinsic namespace in a
 * way that can confuse code paths that expect real AVX. Restrict the
 * include to specific .c/.cpp files that need it.
 *
 * On AVX-capable hosts we intentionally do NOT include the SIMDe
 * headers — the compiler's real intrinsic headers are strictly
 * preferable there. A TU that wants "SIMDe on Westmere, real AVX
 * elsewhere" should wrap its include with:
 *
 *    #if !defined(__AVX__) && defined(__SSE4_1__)
 *    #include "arch/x86/simde-shim.h"
 *    #endif
 *
 * and then use `_mm256_*` intrinsics freely.
 */

#ifndef GGML_ARCH_X86_SIMDE_SHIM_H
#define GGML_ARCH_X86_SIMDE_SHIM_H

#if !defined(__SSE4_1__)
#error "simde-shim.h requires SSE4.1 support (Penryn+ / Westmere+)"
#endif

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#error "simde-shim.h should not be included when the target has native AVX; use the real intrinsics instead"
#endif

/* Provide _mm256_*, _mm_cvtph_ps, _mm_cvtps_ph, etc. as first-class
 * names so that source code written against the AVX/AVX2/F16C intrinsic
 * API compiles unchanged. SIMDe emits SSE4.1-compatible expansions. */
#define SIMDE_ENABLE_NATIVE_ALIASES
#define SIMDE_X86_AVX_NATIVE_ALIASES
#define SIMDE_X86_AVX2_NATIVE_ALIASES
#define SIMDE_X86_FMA_NATIVE_ALIASES
#define SIMDE_X86_F16C_NATIVE_ALIASES

/* Prevent any upstream immintrin.h pull-in from inside SIMDe; SIMDe
 * provides its own declarations and real immintrin.h would conflict. */
#include <simde/x86/avx.h>
#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>
#include <simde/x86/f16c.h>

#endif /* GGML_ARCH_X86_SIMDE_SHIM_H */
