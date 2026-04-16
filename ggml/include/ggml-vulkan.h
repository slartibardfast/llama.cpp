#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_vk_init(size_t dev_num);

GGML_BACKEND_API bool ggml_backend_is_vk(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_vk_get_device_count(void);
GGML_BACKEND_API void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);

// Override the TURBO_*B weight codebook on all initialized Vulkan devices.
// bits must be in [2,5]; centroids points to 2^bits floats. Used when a
// model GGUF carries "turbo.codebook.Nbit" tensors (e.g., per-model
// imatrix-weighted Lloyd-Max centroids from the llama-turbo-codebook tool).
// If no Vulkan devices are initialized yet, the override is ignored; the
// model loader should call this after device init.
GGML_BACKEND_API void ggml_backend_vk_set_turbo_codebook(int bits, const float * centroids);

#ifdef  __cplusplus
}
#endif
