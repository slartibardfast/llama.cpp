// Instantiations for K=F16, V=IQ4_NL fattn-vec kernels.
// Default --cache-type-v iq4_nl path (PROFILING.md). Hand-written
// (not via generate_cu_files.py) since it sits outside the main quant family.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16, GGML_TYPE_IQ4_NL);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16, GGML_TYPE_IQ4_NL);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_F16, GGML_TYPE_IQ4_NL);
