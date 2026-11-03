// radish (Adreno 506) FP32 GEMM shader shared by Convolution1D (1x1) and
// Deconvolution (GEMM stage before col2im). See shader/convolution1d_radish_gemm.comp.
#ifndef LAYER_RADISH_GEMM_H
#define LAYER_RADISH_GEMM_H

#include "gpu.h"
#include "option.h"
#include <stdlib.h>
#include <string.h>

namespace ncnn {

// RADISH_NCNN_GEMM=1 on the Radish A506 with plain FP32 storage and math.
static inline bool radish_gemm_available(const VulkanDevice* vkdev, const Option& opt)
{
    const char* env = getenv("RADISH_NCNN_GEMM");
    return env && strcmp(env, "1") == 0
           && vkdev->info.vendor_id() == 0x5143 && vkdev->info.device_id() == 506
           && strstr(vkdev->info.device_name(), "Radish") != 0
           && vkdev->info.max_shared_memory_size() >= 16384
           && vkdev->info.max_workgroup_invocations() >= 128
           && opt.use_packing_layout
           && !opt.use_fp16_packed && !opt.use_fp16_storage && !opt.use_fp16_arithmetic && !opt.use_fp16_uniform
           && !opt.use_bf16_packed && !opt.use_bf16_storage
           && !opt.use_int8_inference && !opt.use_int8_packed && !opt.use_int8_storage
           && !opt.use_cooperative_matrix && !opt.use_tensor_storage;
}

static inline int radish_gemm_round(int v, int m)
{
    return (v + m - 1) / m * m;
}

} // namespace ncnn

#endif
