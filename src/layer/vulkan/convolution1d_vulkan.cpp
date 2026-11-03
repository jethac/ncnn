// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convolution1d_vulkan.h"
#include "radish_gemm.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <new>
#include <cmath>

#include "allocator.h"
#include "layer_shader_type.h"
#include "layer_type.h"

namespace ncnn {

static bool radish_planar_fp32(const Option& opt)
{
    return opt.use_packing_layout && !opt.use_fp16_packed && !opt.use_fp16_storage
           && !opt.use_fp16_arithmetic && !opt.use_fp16_uniform
           && !opt.use_bf16_packed && !opt.use_bf16_storage
           && !opt.use_int8_inference && !opt.use_int8_packed && !opt.use_int8_storage
           && !opt.use_int8_arithmetic && !opt.use_int8_uniform
           && !opt.use_int16_packed && !opt.use_int16_storage
           && !opt.use_subgroup_ops && !opt.use_cooperative_matrix && !opt.use_tensor_storage;
}

static bool radish_planar_buffer(const VkMat& blob, size_t required, const VulkanDevice* device)
{
    if (!required || !blob.data || !blob.refcount || !blob.allocator
        || blob.allocator->vkdev != device || !blob.data->buffer || !blob.data->memory
        || blob.n != 1 || blob.c <= 0 || blob.elempack <= 0
        || blob.elemsize != (size_t)blob.elempack * 4u
        || blob.cstep > SIZE_MAX / (size_t)blob.c)
        return false;

    const size_t count = blob.cstep * (size_t)blob.c;
    if (!count || count > SIZE_MAX / blob.elemsize)
        return false;
#if NCNN_BATCH
    if (blob.nstep != count)
        return false;
#endif
    const size_t bytes = count * blob.elemsize;
    if (required > bytes || blob.offset > blob.data->capacity
        || bytes > blob.data->capacity - blob.offset
        || blob.data->offset > SIZE_MAX - blob.data->capacity
        || blob.buffer_capacity() != bytes)
        return false;
    const size_t alignment = device->info.buffer_offset_alignment();
    return alignment && blob.buffer_offset() % alignment == 0
           && blob.buffer_offset() % 16 == 0
           && bytes <= device->info.physicalDeviceProperties().limits.maxStorageBufferRange;
}

static bool radish_planar_rank2(const VkMat& blob, int w, int h, int pack)
{
    if (w <= 0 || h <= 0 || (size_t)w > SIZE_MAX / (size_t)h)
        return false;
    return blob.dims == 2 && blob.n == 1 && blob.w == w && blob.h == h
           && blob.d == 1 && blob.c == 1 && blob.elempack == pack
           && blob.elemsize == (size_t)pack * 4u && blob.cstep == (size_t)w * h;
}

static bool radish_planar_disjoint(const VkMat& a, const VkMat& b)
{
    if (a.buffer() != b.buffer())
        return true;
    const size_t ao = a.buffer_offset();
    const size_t bo = b.buffer_offset();
    return ao <= bo ? a.buffer_capacity() <= bo - ao : b.buffer_capacity() <= ao - bo;
}

static bool radish_planar_pipeline(const Pipeline* pipeline, int bindings, int pushes, uint32_t x, uint32_t y)
{
    if (!pipeline || !pipeline->pipeline() || pipeline->local_size_x() != x
        || pipeline->local_size_y() != y || pipeline->local_size_z() != 1
        || pipeline->shader_info().binding_count != bindings
        || pipeline->shader_info().push_constant_count != pushes)
        return false;
    for (int i = 0; i < bindings; i++)
        if (pipeline->shader_info().binding_types[i] != 1)
            return false;
    return true;
}

static int radish_planar_pack_weights(const Mat& original, Mat& packed, int num_input, int padded_input, int num_output, int kernel_w, int weight_data_size, const VulkanDevice* device)
{
    if (num_input <= 0 || padded_input < num_input || num_output <= 0
        || num_output % 64 || kernel_w <= 0 || weight_data_size <= 0)
        return -1;

    const uint64_t original_k = (uint64_t)num_input * kernel_w;
    const uint64_t padded_k = (uint64_t)padded_input * kernel_w;
    if (padded_k > 3584 || padded_k % 32)
        return -1;
    const uint64_t original_count = original_k * num_output;
    const uint64_t padded_count = padded_k * num_output;
    const uint64_t byte_limit = std::min((uint64_t)SIZE_MAX, (uint64_t)device->info.physicalDeviceProperties().limits.maxStorageBufferRange);
    if (original_count != (uint64_t)weight_data_size || padded_count > INT_MAX
        || padded_count > (SIZE_MAX - sizeof(int)) / sizeof(float)
        || padded_count > byte_limit / sizeof(float)
        || original.empty() || original.dims != 1 || original.n != 1
        || original.w != weight_data_size || original.h != 1 || original.d != 1 || original.c != 1
        || original.elemsize != 4u || original.elempack != 1
        || original.cstep < (size_t)original_count)
        return -1;
#if NCNN_BATCH
    if (original.nstep != original.total())
        return -1;
#endif

    Mat destination((int)padded_count, (size_t)4u, 1);
    if (destination.empty())
        return -100;
    destination.fill(0.f);
    const float* source = original;
    float* target = destination;
    for (size_t k = 0; k < (size_t)original_k; k++)
        for (int co = 0; co < num_output; co++)
            memcpy(target + k * num_output + co, source + (size_t)co * original_k + k, sizeof(float));

    packed = destination;
    return 0;
}

static uint16_t radish_weight_half_rte(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 255u;
    const uint32_t fraction = bits & 0x7fffffu;
    if (exponent < 102)
        return (uint16_t)sign;
    if (exponent < 113)
    {
        const uint32_t mantissa = fraction | 0x800000u;
        const unsigned shift = 126 - exponent;
        uint32_t rounded = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const uint32_t midpoint = 1u << (shift - 1);
        rounded += remainder > midpoint || (remainder == midpoint && (rounded & 1u));
        return (uint16_t)(sign | rounded);
    }
    if (exponent > 142)
        return (uint16_t)(sign | 0x7c00u);
    uint32_t rounded = ((exponent - 112) << 10) | (fraction >> 13);
    const uint32_t remainder = fraction & 8191u;
    rounded += remainder > 4096u || (remainder == 4096u && (rounded & 1u));
    return (uint16_t)(sign | rounded);
}

static int radish_direct_pack_half_weights(Mat& packed, int dilation)
{
    const int count = 128 * 128 * 11;
    if (packed.empty() || packed.dims != 1 || packed.n != 1 || packed.w != count
        || packed.elemsize != 4u || packed.elempack != 1 || packed.total() < (size_t)count)
        return -1;
    Mat words(count / 2, (size_t)4u, 1);
    if (words.empty())
        return -100;
    const float* source = packed;
    uint32_t* destination = words;
    double row_norm[128] = {};
    double row_error[128] = {};
    for (int i = 0; i < count; i += 2)
    {
        uint16_t encoded[2];
        for (int j = 0; j < 2; j++)
        {
            uint32_t raw;
            memcpy(&raw, source + i + j, sizeof(raw));
            if ((raw & 0x7f800000u) == 0x7f800000u)
                return 0;
            encoded[j] = radish_weight_half_rte(source[i + j]);
            if ((encoded[j] & 0x7c00u) == 0x7c00u)
                return 0;
            const float decoded = float16_to_float32(encoded[j]);
            const int co = (i + j) & 127;
            row_norm[co] += std::fabs((double)source[i + j]);
            row_error[co] += std::fabs((double)source[i + j] - (double)decoded);
        }
        destination[i / 2] = (uint32_t)encoded[0] | ((uint32_t)encoded[1] << 16);
    }
    for (int co = 0; co < 128; co++)
        if (row_error[co] > row_norm[co] / 4096.0 + (128 * 11) / 33554432.0
            || row_error[co] > 1.0 / 512.0)
            return 0;
    packed = words;
    fprintf(stderr, "RADISH_CONV_PRECISION channels=128 kernel=11 dilation=%d weights=fp16-rte activations=fp32 shared=fp32 accumulation=fp32 bias=fp32 output=fp32 normalization=fp32\n", dilation);
    return 1;
}

static bool radish_pad_channels_tuple(int input, int output)
{
    return (input == 514 && output == 1024)
           || (input == 1090 && (output == 1024 || output == 512));
}

static bool radish_pad_channels_rank2(const VkMat& blob, int w, int h, int pack)
{
    if (w <= 0 || h <= 0 || (pack != 1 && pack != 4)
        || (size_t)w > SIZE_MAX / (size_t)h)
        return false;
    const size_t count = (size_t)w * h;
    const size_t elemsize = (size_t)pack * 4u;
    if (count > (SIZE_MAX - 15) / elemsize)
        return false;
    const size_t aligned_bytes = (count * elemsize + 15) / 16 * 16;
    return blob.dims == 2 && blob.n == 1 && blob.w == w && blob.h == h
           && blob.d == 1 && blob.c == 1 && blob.elempack == pack
           && blob.elemsize == elemsize && blob.cstep == aligned_bytes / elemsize;
}

Convolution1D_vulkan::Convolution1D_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    padding = 0;

    pipeline_convolution1d = 0;
    pipeline_convolution1d_1x1s1d1 = 0;
    pipeline_convolution1d_gemm = 0;
    pipeline_radish_planar_pad = 0;

    use_radish_tiled = false;
    use_radish_gemm = false;
    radish_tiled32 = false;
    radish_planar = false;
    radish_pad_channels = false;
    radish_direct_window = false;
    radish_direct_weight_half = false;

    use_cooperative_matrix = false;
    coopmat_M = 0;
    coopmat_N = 0;
    coopmat_K = 0;
    coopmat_subgroup_size = 0;
    UNROLL_SG_M = 1;
    UNROLL_SG_N = 1;
    UNROLL_SG_K = 1;
    UNROLL_WG_M = 1;
    UNROLL_WG_N = 1;
}

int Convolution1D_vulkan::load_param(const ParamDict& pd)
{
    int ret = Convolution1D::load_param(pd);

    if (dynamic_weight)
    {
        support_vulkan = false;
    }

    return ret;
}

int Convolution1D_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;

    const int maxk = kernel_w;
    int num_input = weight_data_size / maxk / num_output;

    int elempack = num_input % 4 == 0 ? 4 : 1;
    int out_elempack = num_output % 4 == 0 ? 4 : 1;

    {
        padding = ncnn::create_layer_vulkan(ncnn::LayerType::Padding);
        padding->vkdev = vkdev;

        const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
        if (shape.dims == 2 && shape.w > 0 && shape.h > 0)
        {
            long long bordered_w = 0;
            if (pad_left >= 0 && pad_right >= 0)
            {
                bordered_w = (long long)shape.w + pad_left + pad_right;
            }
            else if (((pad_left == -233 && pad_right == -233) || (pad_left == -234 && pad_right == -234))
                     && kernel_w > 0 && dilation_w > 0 && stride_w > 0)
            {
                const long long kernel_extent_w = (long long)dilation_w * (kernel_w - 1) + 1;
                const long long wpad = kernel_extent_w + (shape.w - 1) / stride_w * (long long)stride_w - shape.w;
                bordered_w = shape.w + std::max(0LL, wpad);
            }

            if (bordered_w > 0 && bordered_w <= INT_MAX)
            {
                padding->bottom_shapes.assign(1, shape);
                padding->top_shapes.assign(1, Mat((int)bordered_w, shape.h, (void*)0, shape.elemsize, shape.elempack));
            }
        }

        ncnn::ParamDict pd;
        pd.set(0, 0);
        pd.set(1, 0);
        pd.set(2, pad_left);
        pd.set(3, pad_right);
        pd.set(4, 0);
        pd.set(5, pad_value);

        padding->load_param(pd);

        padding->create_pipeline(opt);
    }

    bool is_conv1x1s1d1 = kernel_w == 1 && stride_w == 1 && dilation_w == 1;

    bool use_gemm = opt.use_sgemm_convolution
                    && !is_conv1x1s1d1
                    && num_input * maxk >= 8
                    && num_output >= 8;

    const char* tiled_env = getenv("RADISH_NCNN_TILED");
    const char* tiled32_env = getenv("RADISH_NCNN_TILED32");
    const bool requested_tiled32 = tiled32_env && strcmp(tiled32_env, "1") == 0;
    use_radish_tiled = (requested_tiled32 || (tiled_env && strcmp(tiled_env, "1") == 0)) && use_gemm
                       && vkdev->info.vendor_id() == 0x5143 && vkdev->info.device_id() == 506
                       && strcmp(vkdev->info.device_name(), "Radish Adreno 506 (experimental compute)") == 0
                       && vkdev->info.max_shared_memory_size() >= 16384
                       && vkdev->info.max_workgroup_invocations() >= 128
                       && vkdev->info.max_workgroup_size_x() >= 16 && vkdev->info.max_workgroup_size_y() >= 8
                       && opt.use_packing_layout && elempack == 4 && out_elempack == 4
                       && stride_w == 1 && (dilation_w == 1 || dilation_w == 3 || dilation_w == 5)
                       && (num_input * maxk) % 32 == 0
                       && num_input * maxk <= 3584
                       && num_output % 64 == 0 && activation_type == 0
                       && !opt.use_fp16_packed && !opt.use_fp16_storage
                       && !opt.use_fp16_arithmetic && !opt.use_fp16_uniform
                       && !opt.use_bf16_packed && !opt.use_bf16_storage
                       && !opt.use_int8_inference && !opt.use_int8_packed && !opt.use_int8_storage
                       && !opt.use_int8_arithmetic && !opt.use_int8_uniform
                       && !opt.use_int16_packed && !opt.use_int16_storage
                       && !opt.use_subgroup_ops && !opt.use_cooperative_matrix && !opt.use_tensor_storage;
    radish_tiled32 = use_radish_tiled && requested_tiled32;
    const char* planar_env = getenv("RADISH_NCNN_PLANAR");
    const char* packed_a_env = getenv("RADISH_NCNN_PLANAR_PACKED_A");
    const bool requested_packed_a = packed_a_env && strcmp(packed_a_env, "1") == 0;
    uint32_t pad_bits;
    memcpy(&pad_bits, &pad_value, sizeof(pad_bits));
    radish_planar = planar_env && strcmp(planar_env, "1") == 0
                    && use_radish_tiled && !radish_tiled32
                    && pad_left >= 0 && pad_right >= 0
                    && (pad_left > 0 || pad_right > 0) && pad_bits == 0;
    radish_direct_window = radish_planar && requested_packed_a
                           && num_input == 128 && num_output == 128 && kernel_w == 11
                           && bias_term == 1 && !dynamic_weight;

    const char* pad_channels_env = getenv("RADISH_NCNN_PAD_CHANNELS");
    radish_pad_channels = pad_channels_env && strcmp(pad_channels_env, "1") == 0
                          && tiled_env && strcmp(tiled_env, "1") == 0
                          && planar_env && strcmp(planar_env, "1") == 0
                          && !requested_tiled32 && use_gemm && !dynamic_weight
                          && vkdev->info.vendor_id() == 0x5143 && vkdev->info.device_id() == 506
                          && strcmp(vkdev->info.device_name(), "Radish Adreno 506 (experimental compute)") == 0
                          && vkdev->info.max_shared_memory_size() >= 16384
                          && vkdev->info.max_workgroup_invocations() >= 128
                          && vkdev->info.max_workgroup_size_x() >= 16 && vkdev->info.max_workgroup_size_y() >= 8
                          && opt.use_vulkan_compute && radish_planar_fp32(opt) && !opt.use_shader_local_memory
                          && elempack == 1 && out_elempack == 4
                          && kernel_w == 3 && dilation_w == 1 && stride_w == 1
                          && pad_left == 1 && pad_right == 1 && pad_bits == 0
                          && bias_term == 1 && activation_type == 0
                          && radish_pad_channels_tuple(num_input, num_output)
                          && (long long)num_input * 3 * num_output == weight_data_size;
    if (radish_pad_channels)
    {
        const int channels = (num_input + 31) / 32 * 32;
        const int k = channels * 3;
        const uint64_t padded_count = (uint64_t)num_output * k;
        const uint64_t byte_limit = std::min((uint64_t)SIZE_MAX, (uint64_t)vkdev->info.physicalDeviceProperties().limits.maxStorageBufferRange);
        if (k <= 0 || k > 3584 || k % 32 || num_output % 64
            || padded_count > INT_MAX || padded_count > byte_limit / 4
            || weight_data.empty() || weight_data.elemsize != 4u || weight_data.elempack != 1
            || weight_data.total() < (size_t)weight_data_size
            || bias_data.empty() || bias_data.elemsize != 4u || bias_data.elempack != 1
            || bias_data.total() < (size_t)num_output)
            return -1;
        const float* original = weight_data;
        const float* bias = bias_data;
        for (int i = 0; i < weight_data_size; i++)
            if (!std::isfinite(original[i]))
                return -1;
        for (int i = 0; i < num_output; i++)
            if (!std::isfinite(bias[i]))
                return -1;
        if (requested_packed_a)
        {
            const int pack_ret = radish_planar_pack_weights(weight_data, weight_data_packed, num_input, channels, num_output, kernel_w, weight_data_size, vkdev);
            if (pack_ret != 0)
                return pack_ret;
        }
        else
        {
            weight_data_packed.create((int)padded_count, (size_t)4u, 1);
            if (weight_data_packed.empty())
                return -100;
            weight_data_packed.fill(0.f);
            float* padded = weight_data_packed;
            const size_t original_row = (size_t)num_input * 3;
            for (int co = 0; co < num_output; co++)
                memcpy(padded + (size_t)co * k, original + (size_t)co * original_row, original_row * sizeof(float));
        }

        use_radish_tiled = true;
        radish_planar = true;
        std::vector<vk_specialization_type> specializations(5);
        specializations[0].i = 3;
        specializations[1].i = channels;
        specializations[2].i = num_output;
        specializations[3].i = 1;
        specializations[4].i = 1;
        pipeline_convolution1d_gemm = new (std::nothrow) Pipeline(vkdev);
        pipeline_radish_planar_pad = new (std::nothrow) Pipeline(vkdev);
        if (!pipeline_convolution1d_gemm || !pipeline_radish_planar_pad)
            return -100;
        pipeline_convolution1d_gemm->set_local_size_xyz(16, 8, 1);
        pipeline_radish_planar_pad->set_local_size_xyz(8, 8, 1);
        const int math_shader_type = requested_packed_a ? LayerShaderType::convolution1d_radish_planar_packed_a : LayerShaderType::convolution1d_radish_planar;
        const int math_ret = pipeline_convolution1d_gemm->create(math_shader_type, opt, specializations);
        if (math_ret != 0)
            return math_ret;
        const std::vector<vk_specialization_type> pad_specializations;
        const int pad_ret = pipeline_radish_planar_pad->create(LayerShaderType::convolution1d_radish_planar_pack1_pad, opt, pad_specializations);
        if (pad_ret != 0)
            return pad_ret;
        if (!radish_planar_pipeline(pipeline_convolution1d_gemm, 4, 2, 16, 8)
            || pipeline_convolution1d_gemm->shader_info().specialization_count != 5
            || !radish_planar_pipeline(pipeline_radish_planar_pad, 2, 5, 8, 8)
            || pipeline_radish_planar_pad->shader_info().specialization_count != 0)
            return -1;
        if (opt.lightmode)
            weight_data.release();
        return 0;
    }

    if (use_radish_tiled)
    {
        if (radish_planar && requested_packed_a)
        {
            const int pack_ret = radish_planar_pack_weights(weight_data, weight_data_packed, num_input, num_input, num_output, kernel_w, weight_data_size, vkdev);
            if (pack_ret != 0)
                return pack_ret;
            if (radish_direct_window)
            {
                const int half_ret = radish_direct_pack_half_weights(weight_data_packed, dilation_w);
                if (half_ret < 0)
                    return half_ret;
                radish_direct_weight_half = half_ret == 1;
                if (!radish_direct_weight_half)
                    fprintf(stderr, "RADISH_CONV_PRECISION channels=128 kernel=11 dilation=%d weights=fp32 activations=fp32 shared=fp32 accumulation=fp32 bias=fp32 output=fp32 normalization=fp32\n", dilation_w);
            }
        }
        else
        {
            // The original weights already have the required [output][input*tap]
            // order. Keep one reference until upload instead of a second packing.
            weight_data_packed = weight_data;
        }

        std::vector<vk_specialization_type> specializations(5);
        specializations[0].i = kernel_w;
        specializations[1].i = num_input;
        specializations[2].i = num_output;
        specializations[3].i = bias_term;
        specializations[4].i = dilation_w;

        pipeline_convolution1d_gemm = new Pipeline(vkdev);
        const uint32_t local_x = radish_tiled32 ? 8 : 16;
        const uint32_t local_y = radish_tiled32 ? 4 : 8;
        pipeline_convolution1d_gemm->set_local_size_xyz(local_x, local_y, 1);
        if (pipeline_convolution1d_gemm->local_size_x() != local_x
            || pipeline_convolution1d_gemm->local_size_y() != local_y
            || pipeline_convolution1d_gemm->local_size_z() != 1)
            return -1;

        const int shader_type = radish_direct_window ? (radish_direct_weight_half ? LayerShaderType::convolution1d_radish_direct_window_fp16w : LayerShaderType::convolution1d_radish_direct_window)
                                  : radish_planar ? (requested_packed_a ? LayerShaderType::convolution1d_radish_planar_packed_a : LayerShaderType::convolution1d_radish_planar)
                                                 : radish_tiled32 ? LayerShaderType::convolution1d_radish_tiled32 : LayerShaderType::convolution1d_radish_tiled;
        const int ret = pipeline_convolution1d_gemm->create(shader_type, opt, specializations);
        if (ret != 0)
            return ret;

        if (radish_direct_window)
        {
            if (!radish_planar_pipeline(pipeline_convolution1d_gemm, 4, 3, 16, 8)
                || pipeline_convolution1d_gemm->shader_info().specialization_count != 5)
                return -1;
        }
        else if (radish_planar)
        {
            if (!radish_planar_pipeline(pipeline_convolution1d_gemm, 4, 2, 16, 8))
                return -1;
            pipeline_radish_planar_pad = new (std::nothrow) Pipeline(vkdev);
            if (!pipeline_radish_planar_pad)
                return -100;
            pipeline_radish_planar_pad->set_local_size_xyz(8, 8, 1);
            const std::vector<vk_specialization_type> pad_specializations;
            const int pad_ret = pipeline_radish_planar_pad->create(LayerShaderType::convolution1d_radish_planar_pad, opt, pad_specializations);
            if (pad_ret != 0)
                return pad_ret;
            if (!radish_planar_pipeline(pipeline_radish_planar_pad, 2, 4, 8, 8))
                return -1;
        }
    }
    else if (use_gemm)
    {
        use_cooperative_matrix = vkdev->info.support_cooperative_matrix() && opt.use_cooperative_matrix && (opt.use_fp16_storage || opt.use_fp16_packed);

        if (use_cooperative_matrix)
        {
            int size = 1024;

            vkdev->info.get_optimal_cooperative_matrix_mnk(size, num_output, num_input * maxk, VK_COMPONENT_TYPE_FLOAT16_KHR, opt.use_fp16_arithmetic ? VK_COMPONENT_TYPE_FLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K, coopmat_subgroup_size);

            if (coopmat_M == 0)
            {
                use_cooperative_matrix = false;
            }
            else
            {
                UNROLL_SG_M = std::min((size + coopmat_M - 1) / coopmat_M, 2);
                UNROLL_SG_N = std::min((num_output + coopmat_N - 1) / coopmat_N, 2);
                UNROLL_SG_K = std::min((num_input * maxk + coopmat_K - 1) / coopmat_K, 2);

                if (vkdev->info.vendor_id() == 0x5143)
                {
                    // adreno driver fails with multiple M tiles per subgroup
                    UNROLL_SG_M = 1;
                    // adreno bias initialization also fails with N unrolling
                    if (bias_term)
                        UNROLL_SG_N = 1;
                }

                UNROLL_WG_M = std::min((size + coopmat_M * UNROLL_SG_M - 1) / (coopmat_M * UNROLL_SG_M), 2);
                UNROLL_WG_N = std::min((num_output + coopmat_N * UNROLL_SG_N - 1) / (coopmat_N * UNROLL_SG_N), 2);

                const size_t shared_a = 2 * coopmat_M * coopmat_K;
                const size_t shared_b = 2 * coopmat_K * coopmat_N;
                const size_t shared_o = 2 * coopmat_M * coopmat_N;

                for (;;)
                {
                    const size_t shared_bytes = shared_a * UNROLL_WG_M * UNROLL_SG_M * UNROLL_SG_K
                                                + shared_b * UNROLL_WG_N * UNROLL_SG_N * UNROLL_SG_K
                                                + shared_o * UNROLL_WG_M * UNROLL_WG_N * UNROLL_SG_M * UNROLL_SG_N;
                    const uint32_t invocations = coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N;
                    if (shared_bytes <= vkdev->info.max_shared_memory_size() && invocations <= vkdev->info.max_workgroup_invocations()
                            && invocations <= vkdev->info.max_workgroup_size_x() && (uint32_t)(UNROLL_WG_M * UNROLL_WG_N) <= vkdev->info.max_compute_workgroup_subgroups())
                        break;

                    // reduce K first to preserve output reuse
                    if (UNROLL_SG_K > 1)
                        UNROLL_SG_K = 1;
                    else if (UNROLL_WG_N > 1)
                        UNROLL_WG_N = 1;
                    else if (UNROLL_WG_M > 1)
                        UNROLL_WG_M = 1;
                    else if (UNROLL_SG_N > 1)
                        UNROLL_SG_N = 1;
                    else if (UNROLL_SG_M > 1)
                        UNROLL_SG_M = 1;
                    else
                    {
                        use_cooperative_matrix = false;
                        break;
                    }
                }
            }
        }

        if (use_cooperative_matrix)
        {
            Mat weight_data_r2;

            if (elempack == 4)
            {
                // from maxk-inch-outch to 4-maxk-inch/4-outch
                weight_data_r2.create(4 * maxk * (num_input / 4) * num_output);
                for (int i = 0; i < num_output; i++)
                {
                    for (int j = 0; j < num_input / 4; j++)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            weight_data_r2[((i * (num_input / 4) + j) * maxk + k) * 4] = weight_data[(i * num_input + j * 4) * maxk + k];
                            weight_data_r2[((i * (num_input / 4) + j) * maxk + k) * 4 + 1] = weight_data[(i * num_input + j * 4 + 1) * maxk + k];
                            weight_data_r2[((i * (num_input / 4) + j) * maxk + k) * 4 + 2] = weight_data[(i * num_input + j * 4 + 2) * maxk + k];
                            weight_data_r2[((i * (num_input / 4) + j) * maxk + k) * 4 + 3] = weight_data[(i * num_input + j * 4 + 3) * maxk + k];
                        }
                    }
                }
            }
            else
            {
                weight_data_r2 = weight_data;
            }

            const int blocks_n = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);
            const int kk = (num_input * maxk + coopmat_K - 1) / coopmat_K;

            weight_data_packed.create(coopmat_N * coopmat_K * UNROLL_SG_N * UNROLL_WG_N * kk, blocks_n);
            for (int bn = 0; bn < blocks_n; bn++)
            {
                float* p = weight_data_packed.row(bn);

                int k = 0;
                for (; k + UNROLL_SG_K - 1 < kk; k += UNROLL_SG_K)
                {
                    for (int wn = 0; wn < UNROLL_WG_N; wn++)
                    {
                        for (int zk = 0; zk < UNROLL_SG_K; zk++)
                        {
                            for (int zn = 0; zn < UNROLL_SG_N; zn++)
                            {
                                for (int i = 0; i < coopmat_K; i++)
                                {
                                    for (int j = 0; j < coopmat_N; j++)
                                    {
                                        const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j;
                                        const int gki = (k + zk) * coopmat_K + i;

                                        if (gni < num_output && gki < num_input * maxk)
                                        {
                                            *p++ = weight_data_r2[gni * num_input * maxk + gki];
                                        }
                                        else
                                        {
                                            *p++ = 0.f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                for (; k < kk; k++)
                {
                    for (int wn = 0; wn < UNROLL_WG_N; wn++)
                    {
                        for (int zn = 0; zn < UNROLL_SG_N; zn++)
                        {
                            for (int i = 0; i < coopmat_K; i++)
                            {
                                for (int j = 0; j < coopmat_N; j++)
                                {
                                    const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j;
                                    const int gki = k * coopmat_K + i;

                                    if (gni < num_output && gki < num_input * maxk)
                                    {
                                        *p++ = weight_data_r2[gni * num_input * maxk + gki];
                                    }
                                    else
                                    {
                                        *p++ = 0.f;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            std::vector<vk_specialization_type> specializations(20 + 4);
            specializations[0].u32 = kernel_w;
            specializations[1].u32 = dilation_w;
            specializations[2].u32 = stride_w;
            specializations[3].i = bias_term;
            specializations[4].i = activation_type;
            specializations[5].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
            specializations[6].f = activation_params.w == 2 ? activation_params[1] : 0.f;
            specializations[7].u32 = coopmat_M;
            specializations[8].u32 = coopmat_N;
            specializations[9].u32 = coopmat_K;
            specializations[10].u32 = coopmat_subgroup_size;
            specializations[11].u32 = UNROLL_SG_M;
            specializations[12].u32 = UNROLL_SG_N;
            specializations[13].u32 = UNROLL_SG_K;
            specializations[14].u32 = UNROLL_WG_M;
            specializations[15].u32 = UNROLL_WG_N;
            specializations[16].u32 = num_input;
            specializations[17].u32 = num_output;
            specializations[18].u32 = elempack;
            specializations[19].u32 = out_elempack;
            specializations[20 + 0].i = 0; // w
            specializations[20 + 1].i = 0; // cstep
            specializations[20 + 2].i = 0; // outw
            specializations[20 + 3].i = 0; // outcstep

            pipeline_convolution1d_gemm = new Pipeline(vkdev);
            pipeline_convolution1d_gemm->set_subgroup_size(coopmat_subgroup_size);
            pipeline_convolution1d_gemm->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
            pipeline_convolution1d_gemm->create(LayerShaderType::convolution1d_gemm_cm, opt, specializations);
        }
        else
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            const int num_input_packed = (num_input + 3) / 4 * 4;
            const int num_output_packed = (num_output + 3) / 4 * 4;

            weight_data_packed.create(maxk * num_input_packed / 4, num_output_packed / 4, (size_t)4 * 4 * 4, 4 * 4);

            for (int q = 0; q < num_output_packed; q += 4)
            {
                float* g00 = weight_data_packed.row(q / 4);

                for (int p = 0; p < num_input_packed; p += 4)
                {
                    for (int k = 0; k < maxk; k++)
                    {
                        for (int i = 0; i < 4; i++)
                        {
                            for (int j = 0; j < 4; j++)
                            {
                                if (q + i < num_output && p + j < num_input)
                                {
                                    const float* k00 = weight_data_r2.channel(q + i).row(p + j);
                                    g00[0] = k00[k];
                                }
                                else
                                {
                                    g00[0] = 0.f;
                                }
                                g00++;
                            }
                        }
                    }
                }
            }

            std::vector<vk_specialization_type> specializations(9 + 6);
            specializations[0].i = kernel_w;
            specializations[1].i = dilation_w;
            specializations[2].i = stride_w;
            specializations[3].i = bias_term;
            specializations[4].i = activation_type;
            specializations[5].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
            specializations[6].f = activation_params.w == 2 ? activation_params[1] : 0.f;
            specializations[7].i = elempack;
            specializations[8].i = out_elempack;
            specializations[9 + 0].i = 0;
            specializations[9 + 1].i = num_input_packed / 4;
            specializations[9 + 2].i = 0;
            specializations[9 + 3].i = 0;
            specializations[9 + 4].i = num_output;
            specializations[9 + 5].i = num_input;

            Option gemm_opt = opt;
            const char* shared32_env = getenv("RADISH_NCNN_SHARED32");
            const bool use_shared32 = shared32_env && strcmp(shared32_env, "1") == 0
                                      && vkdev->info.vendor_id() == 0x5143 && vkdev->info.device_id() == 506
                                      && strcmp(vkdev->info.device_name(), "Radish Adreno 506 (experimental compute)") == 0
                                      && !opt.use_fp16_packed && !opt.use_fp16_storage
                                      && !opt.use_fp16_arithmetic && !opt.use_fp16_uniform
                                      && !opt.use_bf16_packed && !opt.use_bf16_storage
                                      && !opt.use_int8_inference;
            if (use_shared32)
                gemm_opt.use_shader_local_memory = true;
            if (gemm_opt.use_shader_local_memory)
            {
                specializations.resize(9 + 7);
                specializations[9 + 6].i = use_shared32 ? 1 : 0;
            }

            pipeline_convolution1d_gemm = new Pipeline(vkdev);
            if (use_shared32)
            {
                pipeline_convolution1d_gemm->set_local_size_xyz(8, 4, 1);
            }
            else if (gemm_opt.use_shader_local_memory)
            {
                pipeline_convolution1d_gemm->set_local_size_xyz(8, 8, 1);
            }
            else
            {
                pipeline_convolution1d_gemm->set_local_size_xyz(16, std::min(4, num_output_packed / 4), 1);
            }
            pipeline_convolution1d_gemm->create(LayerShaderType::convolution1d_packed_gemm, gemm_opt, specializations);
        }
    }
    else if (is_conv1x1s1d1 && !dynamic_weight && out_elempack == 4 && activation_type == 0
             && radish_gemm_available(vkdev, opt))
    {
        // radish: 1x1 Conv1D as a 64x64-tiled GEMM over [num_output][num_input]
        use_radish_gemm = true;
        const int Mp = radish_gemm_round(num_output, 64);
        const int Kp = radish_gemm_round(num_input, 32);
        weight_data_packed.create(Kp * Mp, (size_t)4u, 1);
        if (weight_data_packed.empty())
            return -100;
        weight_data_packed.fill(0.f);
        float* a = weight_data_packed;
        const float* wsrc = weight_data;
        for (int co = 0; co < num_output; co++)
            for (int ci = 0; ci < num_input; ci++)
                a[(size_t)co * Kp + ci] = wsrc[(size_t)co * num_input + ci];
        std::vector<vk_specialization_type> specializations(4);
        specializations[0].i = num_input;
        specializations[1].i = Kp;
        specializations[2].i = elempack;
        specializations[3].i = bias_term;
        pipeline_convolution1d_gemm = new Pipeline(vkdev);
        pipeline_convolution1d_gemm->set_local_size_xyz(16, 8, 1);
        pipeline_convolution1d_gemm->create(LayerShaderType::convolution1d_radish_gemm, opt, specializations);
    }
    else if (is_conv1x1s1d1)
    {
        use_cooperative_matrix = vkdev->info.support_cooperative_matrix() && opt.use_cooperative_matrix && (opt.use_fp16_storage || opt.use_fp16_packed) && num_input >= 8 && num_output >= 8;

        if (use_cooperative_matrix)
        {
            int size = 1024;

            vkdev->info.get_optimal_cooperative_matrix_mnk(size, num_output, num_input, VK_COMPONENT_TYPE_FLOAT16_KHR, opt.use_fp16_arithmetic ? VK_COMPONENT_TYPE_FLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K, coopmat_subgroup_size);

            if (coopmat_M == 0)
            {
                use_cooperative_matrix = false;
            }
            else
            {
                UNROLL_SG_M = std::min((size + coopmat_M - 1) / coopmat_M, 2);
                UNROLL_SG_N = std::min((num_output + coopmat_N - 1) / coopmat_N, 2);
                UNROLL_SG_K = std::min((num_input + coopmat_K - 1) / coopmat_K, 2);

                if (vkdev->info.vendor_id() == 0x5143)
                {
                    // adreno driver fails with multiple M tiles per subgroup
                    UNROLL_SG_M = 1;
                    // adreno bias initialization also fails with N unrolling
                    if (bias_term)
                        UNROLL_SG_N = 1;
                }

                UNROLL_WG_M = std::min((size + coopmat_M * UNROLL_SG_M - 1) / (coopmat_M * UNROLL_SG_M), 2);
                UNROLL_WG_N = std::min((num_output + coopmat_N * UNROLL_SG_N - 1) / (coopmat_N * UNROLL_SG_N), 2);

                const size_t shared_a = 2 * coopmat_M * coopmat_K;
                const size_t shared_b = 2 * coopmat_K * coopmat_N;
                const size_t shared_o = 2 * coopmat_M * coopmat_N;

                for (;;)
                {
                    const size_t shared_bytes = shared_a * UNROLL_WG_M * UNROLL_SG_M * UNROLL_SG_K
                                                + shared_b * UNROLL_WG_N * UNROLL_SG_N * UNROLL_SG_K
                                                + shared_o * UNROLL_WG_M * UNROLL_WG_N * UNROLL_SG_M * UNROLL_SG_N;
                    const uint32_t invocations = coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N;
                    if (shared_bytes <= vkdev->info.max_shared_memory_size() && invocations <= vkdev->info.max_workgroup_invocations()
                            && invocations <= vkdev->info.max_workgroup_size_x() && (uint32_t)(UNROLL_WG_M * UNROLL_WG_N) <= vkdev->info.max_compute_workgroup_subgroups())
                        break;

                    // reduce K first to preserve output reuse
                    if (UNROLL_SG_K > 1)
                        UNROLL_SG_K = 1;
                    else if (UNROLL_WG_N > 1)
                        UNROLL_WG_N = 1;
                    else if (UNROLL_WG_M > 1)
                        UNROLL_WG_M = 1;
                    else if (UNROLL_SG_N > 1)
                        UNROLL_SG_N = 1;
                    else if (UNROLL_SG_M > 1)
                        UNROLL_SG_M = 1;
                    else
                    {
                        use_cooperative_matrix = false;
                        break;
                    }
                }
            }
        }

        if (use_cooperative_matrix)
        {
            const int blocks_n = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);
            const int kk = (num_input + coopmat_K - 1) / coopmat_K;

            weight_data_packed.create(coopmat_N * coopmat_K * UNROLL_SG_N * UNROLL_WG_N * kk, blocks_n);
            for (int bn = 0; bn < blocks_n; bn++)
            {
                float* p = weight_data_packed.row(bn);

                int k = 0;
                for (; k + UNROLL_SG_K - 1 < kk; k += UNROLL_SG_K)
                {
                    for (int wn = 0; wn < UNROLL_WG_N; wn++)
                    {
                        for (int zk = 0; zk < UNROLL_SG_K; zk++)
                        {
                            for (int zn = 0; zn < UNROLL_SG_N; zn++)
                            {
                                for (int i = 0; i < coopmat_K; i++)
                                {
                                    for (int j = 0; j < coopmat_N; j++)
                                    {
                                        const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j;
                                        const int gki = (k + zk) * coopmat_K + i;

                                        if (gni < num_output && gki < num_input)
                                        {
                                            *p++ = weight_data[gni * num_input + gki];
                                        }
                                        else
                                        {
                                            *p++ = 0.f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                for (; k < kk; k++)
                {
                    for (int wn = 0; wn < UNROLL_WG_N; wn++)
                    {
                        for (int zn = 0; zn < UNROLL_SG_N; zn++)
                        {
                            for (int i = 0; i < coopmat_K; i++)
                            {
                                for (int j = 0; j < coopmat_N; j++)
                                {
                                    const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j;
                                    const int gki = k * coopmat_K + i;

                                    if (gni < num_output && gki < num_input)
                                    {
                                        *p++ = weight_data[gni * num_input + gki];
                                    }
                                    else
                                    {
                                        *p++ = 0.f;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            std::vector<vk_specialization_type> specializations(17 + 3);
            specializations[0].i = bias_term;
            specializations[1].i = activation_type;
            specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
            specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
            specializations[4].u32 = coopmat_M;
            specializations[5].u32 = coopmat_N;
            specializations[6].u32 = coopmat_K;
            specializations[7].u32 = coopmat_subgroup_size;
            specializations[8].u32 = UNROLL_SG_M;
            specializations[9].u32 = UNROLL_SG_N;
            specializations[10].u32 = UNROLL_SG_K;
            specializations[11].u32 = UNROLL_WG_M;
            specializations[12].u32 = UNROLL_WG_N;
            specializations[13].u32 = num_input;
            specializations[14].u32 = num_output;
            specializations[15].u32 = elempack;
            specializations[16].u32 = out_elempack;
            specializations[17 + 0].u32 = 0; // size
            specializations[17 + 1].u32 = 0; // cstep
            specializations[17 + 2].u32 = 0; // outcstep

            pipeline_convolution1d_1x1s1d1 = new Pipeline(vkdev);
            pipeline_convolution1d_1x1s1d1->set_subgroup_size(coopmat_subgroup_size);
            pipeline_convolution1d_1x1s1d1->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
            pipeline_convolution1d_1x1s1d1->create(LayerShaderType::convolution1d_1x1s1d1_cm, opt, specializations);
        }
        else
        {
            const int num_input_packed = (num_input + 3) / 4 * 4;
            const int num_output_packed = (num_output + 3) / 4 * 4;

            weight_data_packed.create(num_input_packed / 4, num_output_packed / 4, (size_t)4 * 4 * 4, 4 * 4);

            for (int q = 0; q < num_output_packed; q += 4)
            {
                float* g00 = weight_data_packed.row(q / 4);

                for (int p = 0; p < num_input_packed; p += 4)
                {
                    for (int i = 0; i < 4; i++)
                    {
                        for (int j = 0; j < 4; j++)
                        {
                            if (q + i < num_output && p + j < num_input)
                            {
                                g00[0] = weight_data[(q + i) * num_input + (p + j)];
                            }
                            else
                            {
                                g00[0] = 0.f;
                            }
                            g00++;
                        }
                    }
                }
            }

            const int outh_pack4 = num_output_packed / 4;

            std::vector<vk_specialization_type> specializations(6 + 6);
            specializations[0].i = bias_term;
            specializations[1].i = activation_type;
            specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
            specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
            specializations[4].i = elempack;
            specializations[5].i = out_elempack;
            specializations[6 + 0].i = 0; // w
            specializations[6 + 1].i = 0; // h
            specializations[6 + 2].i = 0; // outw
            specializations[6 + 3].i = outh_pack4;
            specializations[6 + 4].i = num_output;
            specializations[6 + 5].i = num_input_packed;

            pipeline_convolution1d_1x1s1d1 = new Pipeline(vkdev);
            pipeline_convolution1d_1x1s1d1->set_local_size_xyz(8, std::min(8, outh_pack4), 1);
            pipeline_convolution1d_1x1s1d1->create(LayerShaderType::convolution1d_packed_1x1s1d1, opt, specializations);
        }
    }
    else
    {
        Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

        const int num_output_packed = (num_output + 3) / 4 * 4;

        weight_data_packed.create(maxk, num_input / elempack, num_output_packed / 4, (size_t)4 * 4 * elempack, 4 * elempack);

        for (int q = 0; q < num_output_packed; q += 4)
        {
            float* g00 = weight_data_packed.channel(q / 4);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < 4; i++)
                    {
                        for (int j = 0; j < elempack; j++)
                        {
                            if (q + i < num_output)
                            {
                                const float* k00 = weight_data_r2.channel(q + i).row(p + j);
                                g00[0] = k00[k];
                            }
                            else
                            {
                                g00[0] = 0.f;
                            }
                            g00++;
                        }
                    }
                }
            }
        }

        std::vector<vk_specialization_type> specializations(9 + 5);
        specializations[0].i = kernel_w;
        specializations[1].i = dilation_w;
        specializations[2].i = stride_w;
        specializations[3].i = bias_term;
        specializations[4].i = activation_type;
        specializations[5].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
        specializations[6].f = activation_params.w == 2 ? activation_params[1] : 0.f;
        specializations[7].i = elempack;
        specializations[8].i = out_elempack;
        specializations[9 + 0].i = 0;
        specializations[9 + 1].i = 0;
        specializations[9 + 2].i = 0;
        specializations[9 + 3].i = 0;
        specializations[9 + 4].i = num_output;

        pipeline_convolution1d = new Pipeline(vkdev);
        pipeline_convolution1d->set_optimal_local_size_xyz(1, 1, 1);
        pipeline_convolution1d->create(LayerShaderType::convolution1d_packed, opt, specializations);
    }

    if (opt.lightmode)
    {
        weight_data.release();
    }

    return 0;
}

int Convolution1D_vulkan::destroy_pipeline(const Option& opt)
{
    if (padding)
    {
        padding->destroy_pipeline(opt);
        delete padding;
        padding = 0;
    }

    delete pipeline_convolution1d;
    pipeline_convolution1d = 0;

    delete pipeline_convolution1d_1x1s1d1;
    pipeline_convolution1d_1x1s1d1 = 0;

    delete pipeline_convolution1d_gemm;
    pipeline_convolution1d_gemm = 0;

    delete pipeline_radish_planar_pad;
    pipeline_radish_planar_pad = 0;

    use_radish_tiled = false;
    use_radish_gemm = false;
    radish_tiled32 = false;
    radish_planar = false;
    radish_pad_channels = false;
    radish_direct_window = false;
    radish_direct_weight_half = false;

    use_cooperative_matrix = false;
    coopmat_M = 0;
    coopmat_N = 0;
    coopmat_K = 0;
    coopmat_subgroup_size = 0;
    UNROLL_SG_M = 1;
    UNROLL_SG_N = 1;
    UNROLL_SG_K = 1;
    UNROLL_WG_M = 1;
    UNROLL_WG_N = 1;

    return 0;
}

int Convolution1D_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (padding)
    {
        padding->upload_model(cmd, opt);
    }

    cmd.record_upload(weight_data_packed, weight_data_gpu, opt);

    weight_data_packed.release();

    if (bias_term)
    {
        cmd.record_upload(bias_data, bias_data_gpu, opt);

        bias_data.release();
    }

    return 0;
}

int Convolution1D_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    if (radish_direct_window)
    {
        uint32_t pad_bits;
        memcpy(&pad_bits, &pad_value, sizeof(pad_bits));
        if (!use_radish_tiled || !radish_planar || radish_tiled32 || radish_pad_channels
            || use_cooperative_matrix || !radish_planar_fp32(opt) || !opt.use_vulkan_compute
            || !radish_planar_pipeline(pipeline_convolution1d_gemm, 4, 3, 16, 8)
            || pipeline_convolution1d_gemm->shader_info().specialization_count != 5
            || pipeline_radish_planar_pad || !opt.blob_vkallocator
            || opt.blob_vkallocator->vkdev != vkdev || &bottom_blob == &top_blob
            || !radish_planar_rank2(bottom_blob, bottom_blob.w, 32, 4)
            || kernel_w != 11 || num_output != 128 || weight_data_size != 180224
            || stride_w != 1 || (dilation_w != 1 && dilation_w != 3 && dilation_w != 5)
            || bias_term != 1 || dynamic_weight || activation_type != 0
            || pad_left < 0 || pad_right < 0 || pad_bits != 0)
            return -1;
        const int64_t outw64 = (int64_t)bottom_blob.w + pad_left + pad_right - 10 * dilation_w;
        const uint64_t rounded = outw64 > 0 ? ((uint64_t)outw64 + 63) / 64 * 64 : 0;
        const uint64_t input_bytes = (uint64_t)bottom_blob.w * 128 * 4;
        const uint64_t output_bytes = outw64 > 0 ? (uint64_t)outw64 * 128 * 4 : 0;
        const VkPhysicalDeviceLimits& limits = vkdev->info.physicalDeviceProperties().limits;
        const uint64_t byte_limit = std::min((uint64_t)SIZE_MAX, (uint64_t)limits.maxStorageBufferRange);
        if (outw64 <= 0 || outw64 > INT_MAX || rounded > INT_MAX - 128
            || (uint64_t)pad_left > INT_MAX - rounded - 128
            || input_bytes > byte_limit || output_bytes > byte_limit
            || input_bytes / 4 > INT_MAX || output_bytes / 4 > INT_MAX
            || rounded / 64 > limits.maxComputeWorkGroupCount[0]
            || limits.maxComputeWorkGroupCount[1] < 2 || !limits.maxComputeWorkGroupCount[2]
            || rounded / 64 > INT_MAX / 16
            || !radish_planar_buffer(bottom_blob, (size_t)input_bytes, vkdev)
            || !radish_planar_buffer(weight_data_gpu, (size_t)weight_data_size * (radish_direct_weight_half ? 2u : 4u), vkdev)
            || !radish_planar_buffer(bias_data_gpu, 128 * 4, vkdev))
            return -1;
        const int outw = (int)outw64;
        VkMat output;
        output.create(outw, 32, (size_t)16u, 4, opt.blob_vkallocator);
        if (output.empty())
            return -100;
        if (!radish_planar_rank2(output, outw, 32, 4)
            || !radish_planar_buffer(output, (size_t)output_bytes, vkdev)
            || !radish_planar_disjoint(output, bottom_blob)
            || !radish_planar_disjoint(output, weight_data_gpu)
            || !radish_planar_disjoint(output, bias_data_gpu))
            return -1;
        std::vector<VkMat> bindings(4);
        bindings[0] = weight_data_gpu;
        bindings[1] = bottom_blob;
        bindings[2] = output;
        bindings[3] = bias_data_gpu;
        std::vector<vk_constant_type> constants(3);
        constants[0].i = bottom_blob.w;
        constants[1].i = outw;
        constants[2].i = pad_left;
        VkMat dispatcher;
        dispatcher.w = (int)(rounded / 64) * 16;
        dispatcher.h = 16;
        dispatcher.c = 1;
        cmd.record_pipeline(pipeline_convolution1d_gemm, bindings, constants, dispatcher);
        top_blob = output;
        return 0;
    }
    if (radish_pad_channels)
    {
        uint32_t pad_bits;
        memcpy(&pad_bits, &pad_value, sizeof(pad_bits));
        if (!use_radish_tiled || !radish_planar || radish_tiled32 || use_cooperative_matrix
            || !opt.use_vulkan_compute || !radish_planar_fp32(opt) || opt.use_shader_local_memory
            || !radish_planar_pipeline(pipeline_convolution1d_gemm, 4, 2, 16, 8)
            || pipeline_convolution1d_gemm->shader_info().specialization_count != 5
            || !radish_planar_pipeline(pipeline_radish_planar_pad, 2, 5, 8, 8)
            || pipeline_radish_planar_pad->shader_info().specialization_count != 0
            || !opt.workspace_vkallocator || !opt.blob_vkallocator
            || opt.workspace_vkallocator->vkdev != vkdev || opt.blob_vkallocator->vkdev != vkdev
            || &bottom_blob == &top_blob || bottom_blob.w <= 0
            || !radish_pad_channels_tuple(bottom_blob.h, num_output)
            || kernel_w != 3 || stride_w != 1 || dilation_w != 1
            || pad_left != 1 || pad_right != 1 || pad_bits != 0
            || bias_term != 1 || dynamic_weight || activation_type != 0)
            return -1;
        const int win = bottom_blob.w;
        const int input_channels = bottom_blob.h;
        const int channels = (input_channels + 31) / 32 * 32;
        const int k = channels * 3;
        if (k <= 0 || k > 3584 || k % 32 || num_output % 64
            || (long long)input_channels * 3 * num_output != weight_data_size)
            return -1;
        const uint64_t rounded_width = ((uint64_t)win + 63) / 64 * 64;
        const uint64_t pitch = (rounded_width + 2 + 3) / 4 * 4;
        const uint64_t input_count = ((uint64_t)input_channels * win + 3) / 4 * 4;
        const uint64_t workspace_count = (uint64_t)channels * pitch;
        const uint64_t output_count = (uint64_t)num_output * win;
        const uint64_t weight_count = (uint64_t)num_output * k;
        const uint64_t bias_count = (uint64_t)num_output;
        const VkPhysicalDeviceLimits& limits = vkdev->info.physicalDeviceProperties().limits;
        const uint64_t byte_limit = std::min((uint64_t)SIZE_MAX, (uint64_t)limits.maxStorageBufferRange);
        if (!pitch || pitch > INT_MAX || input_count > INT_MAX || workspace_count > INT_MAX
            || output_count > INT_MAX || weight_count > INT_MAX
            || input_count > byte_limit / 4 || workspace_count > byte_limit / 4
            || output_count > byte_limit / 4 || weight_count > byte_limit / 4
            || bias_count > byte_limit / 4
            || (uint64_t)(channels - 1) * pitch + rounded_width + 1 >= workspace_count)
            return -1;
        const int lp = (int)pitch;
        if (!radish_pad_channels_rank2(bottom_blob, win, input_channels, 1)
            || !radish_planar_buffer(bottom_blob, (size_t)input_count * 4, vkdev)
            || !radish_planar_buffer(weight_data_gpu, (size_t)weight_count * 4, vkdev)
            || !radish_planar_buffer(bias_data_gpu, (size_t)bias_count * 4, vkdev))
            return -1;
        const uint64_t pad_groups_x = (pitch / 4 + 7) / 8;
        const uint64_t pad_groups_y = ((uint64_t)channels + 7) / 8;
        const uint64_t math_groups_x = rounded_width / 64;
        const uint64_t math_groups_y = (uint64_t)num_output / 64;
        if (pad_groups_x > limits.maxComputeWorkGroupCount[0]
            || pad_groups_y > limits.maxComputeWorkGroupCount[1]
            || math_groups_x > limits.maxComputeWorkGroupCount[0]
            || math_groups_y > limits.maxComputeWorkGroupCount[1]
            || !limits.maxComputeWorkGroupCount[2]
            || math_groups_x > INT_MAX / 16 || math_groups_y > INT_MAX / 8)
            return -1;
        VkMat workspace;
        workspace.create(lp, channels, (size_t)4u, 1, opt.workspace_vkallocator);
        VkMat output;
        output.create(win, num_output / 4, (size_t)16u, 4, opt.blob_vkallocator);
        if (workspace.empty() || output.empty())
            return -100;
        if (!radish_pad_channels_rank2(workspace, lp, channels, 1)
            || !radish_pad_channels_rank2(output, win, num_output / 4, 4)
            || !radish_planar_buffer(workspace, (size_t)workspace_count * 4, vkdev)
            || !radish_planar_buffer(output, (size_t)output_count * 4, vkdev)
            || !radish_planar_disjoint(workspace, bottom_blob)
            || !radish_planar_disjoint(workspace, weight_data_gpu)
            || !radish_planar_disjoint(workspace, bias_data_gpu)
            || !radish_planar_disjoint(output, bottom_blob)
            || !radish_planar_disjoint(output, weight_data_gpu)
            || !radish_planar_disjoint(output, bias_data_gpu)
            || !radish_planar_disjoint(workspace, output))
            return -1;
        std::vector<VkMat> pad_bindings(2);
        pad_bindings[0] = bottom_blob;
        pad_bindings[1] = workspace;
        std::vector<vk_constant_type> pad_constants(5);
        pad_constants[0].i = win;
        pad_constants[1].i = input_channels;
        pad_constants[2].i = channels;
        pad_constants[3].i = lp;
        pad_constants[4].i = 1;
        VkMat pad_dispatcher;
        pad_dispatcher.w = lp / 4;
        pad_dispatcher.h = channels;
        pad_dispatcher.c = 1;
        cmd.record_pipeline(pipeline_radish_planar_pad, pad_bindings, pad_constants, pad_dispatcher);
        std::vector<VkMat> bindings(4);
        bindings[0] = weight_data_gpu;
        bindings[1] = workspace;
        bindings[2] = output;
        bindings[3] = bias_data_gpu;
        std::vector<vk_constant_type> constants(2);
        constants[0].i = lp;
        constants[1].i = win;
        VkMat dispatcher;
        dispatcher.w = (int)math_groups_x * 16;
        dispatcher.h = (int)math_groups_y * 8;
        dispatcher.c = 1;
        cmd.record_pipeline(pipeline_convolution1d_gemm, bindings, constants, dispatcher);
        top_blob = output;
        return 0;
    }

    if (radish_planar)
    {
        uint32_t pad_bits;
        memcpy(&pad_bits, &pad_value, sizeof(pad_bits));
        if (!use_radish_tiled || radish_tiled32 || !radish_planar_fp32(opt)
            || !radish_planar_pipeline(pipeline_convolution1d_gemm, 4, 2, 16, 8)
            || !radish_planar_pipeline(pipeline_radish_planar_pad, 2, 4, 8, 8)
            || !opt.workspace_vkallocator || !opt.blob_vkallocator
            || opt.workspace_vkallocator->vkdev != vkdev || opt.blob_vkallocator->vkdev != vkdev
            || &bottom_blob == &top_blob || bottom_blob.w <= 0 || bottom_blob.h <= 0
            || num_output <= 0 || num_output % 64 || kernel_w <= 0 || stride_w != 1
            || (dilation_w != 1 && dilation_w != 3 && dilation_w != 5)
            || weight_data_size <= 0 || dynamic_weight || activation_type != 0
            || pad_left < 0 || pad_right < 0 || (pad_left == 0 && pad_right == 0)
            || pad_bits != 0)
            return -1;

        const long long input_channels = (long long)bottom_blob.h * 4;
        if (input_channels > 3584 || kernel_w > 3584 / input_channels)
            return -1;
        const long long k = input_channels * kernel_w;
        if (k <= 0 || k > 3584 || k % 32
            || k * num_output != weight_data_size)
            return -1;

        const int win = bottom_blob.w;
        const int channels = (int)input_channels;
        const long long extent = (long long)dilation_w * (kernel_w - 1);
        const long long bordered_width = (long long)win + pad_left + pad_right;
        const long long output_width = bordered_width - extent;
        if (extent < 0 || extent > INT_MAX || bordered_width > INT_MAX
            || output_width <= 0 || output_width > INT_MAX)
            return -1;
        const long long padded_output_width = ((output_width + 63) / 64) * 64;
        const long long pitch = ((padded_output_width + extent + 3) / 4) * 4;
        if (pitch <= 0 || pitch > INT_MAX
            || input_channels * pitch > INT_MAX || input_channels * win > INT_MAX
            || (long long)num_output * output_width > INT_MAX)
            return -1;

        const int lp = (int)pitch;
        const int outw = (int)output_width;
        const uint64_t input_bytes = (uint64_t)input_channels * win * 4;
        const uint64_t workspace_bytes = (uint64_t)input_channels * lp * 4;
        const uint64_t output_bytes = (uint64_t)num_output * outw * 4;
        const uint64_t weight_bytes = (uint64_t)weight_data_size * 4;
        const uint64_t bias_bytes = (uint64_t)num_output * 4;
        const VkPhysicalDeviceLimits& limits = vkdev->info.physicalDeviceProperties().limits;
        const uint64_t byte_limit = std::min((uint64_t)SIZE_MAX, (uint64_t)limits.maxStorageBufferRange);
        if (input_bytes > byte_limit || workspace_bytes > byte_limit || output_bytes > byte_limit
            || weight_bytes > byte_limit || (bias_term && bias_bytes > byte_limit)
            || !radish_planar_rank2(bottom_blob, win, channels / 4, 4)
            || !radish_planar_buffer(bottom_blob, (size_t)input_bytes, vkdev)
            || !radish_planar_buffer(weight_data_gpu, (size_t)weight_bytes, vkdev)
            || (bias_term && !radish_planar_buffer(bias_data_gpu, (size_t)bias_bytes, vkdev)))
            return -1;

        const uint64_t pad_groups_x = ((uint64_t)lp / 4 + 7) / 8;
        const uint64_t pad_groups_y = ((uint64_t)channels + 7) / 8;
        const uint64_t math_groups_x = ((uint64_t)outw + 63) / 64;
        const uint64_t math_groups_y = (uint64_t)num_output / 64;
        if (pad_groups_x > limits.maxComputeWorkGroupCount[0]
            || pad_groups_y > limits.maxComputeWorkGroupCount[1]
            || math_groups_x > limits.maxComputeWorkGroupCount[0]
            || math_groups_y > limits.maxComputeWorkGroupCount[1]
            || !limits.maxComputeWorkGroupCount[2])
            return -1;

        VkMat workspace;
        workspace.create(lp, channels, (size_t)4u, 1, opt.workspace_vkallocator);
        VkMat output;
        output.create(outw, num_output / 4, (size_t)16u, 4, opt.blob_vkallocator);
        if (workspace.empty() || output.empty())
            return -100;
        if (!radish_planar_rank2(workspace, lp, channels, 1)
            || !radish_planar_rank2(output, outw, num_output / 4, 4)
            || !radish_planar_buffer(workspace, (size_t)workspace_bytes, vkdev)
            || !radish_planar_buffer(output, (size_t)output_bytes, vkdev)
            || !radish_planar_disjoint(workspace, bottom_blob)
            || !radish_planar_disjoint(workspace, weight_data_gpu)
            || !radish_planar_disjoint(output, bottom_blob)
            || !radish_planar_disjoint(output, weight_data_gpu)
            || !radish_planar_disjoint(workspace, output)
            || (bias_term && (!radish_planar_disjoint(workspace, bias_data_gpu)
                              || !radish_planar_disjoint(output, bias_data_gpu))))
            return -1;

        std::vector<VkMat> pad_bindings(2);
        pad_bindings[0] = bottom_blob;
        pad_bindings[1] = workspace;
        std::vector<vk_constant_type> pad_constants(4);
        pad_constants[0].i = win;
        pad_constants[1].i = channels;
        pad_constants[2].i = lp;
        pad_constants[3].i = pad_left;
        VkMat pad_dispatcher;
        pad_dispatcher.w = lp / 4;
        pad_dispatcher.h = channels;
        pad_dispatcher.c = 1;
        cmd.record_pipeline(pipeline_radish_planar_pad, pad_bindings, pad_constants, pad_dispatcher);

        std::vector<VkMat> bindings(4);
        bindings[0] = weight_data_gpu;
        bindings[1] = workspace;
        bindings[2] = output;
        bindings[3] = bias_data_gpu;
        std::vector<vk_constant_type> constants(2);
        constants[0].i = lp;
        constants[1].i = outw;
        VkMat dispatcher;
        dispatcher.w = (int)math_groups_x * 16;
        dispatcher.h = (int)math_groups_y * 8;
        dispatcher.c = 1;
        cmd.record_pipeline(pipeline_convolution1d_gemm, bindings, constants, dispatcher);
        top_blob = output;
        return 0;
    }

    int w = bottom_blob.w;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    VkMat bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0)
    {
        Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        padding->forward(bottom_blob, bottom_blob_bordered, cmd, opt_pad);
    }
    else if (pad_left == -233 && pad_right == -233)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        if (wpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = 0;
            padding_params[1] = 0;
            padding_params[2] = wpad / 2;
            padding_params[3] = wpad - wpad / 2;
            padding_params[4] = 0;
            padding_params[5] = 0;
            std::vector<VkMat> padding_inputs(2);
            padding_inputs[0] = bottom_blob;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkMat> padding_outputs(1);
            padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);
            bottom_blob_bordered = padding_outputs[0];
        }
    }
    else if (pad_left == -234 && pad_right == -234)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        if (wpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = 0;
            padding_params[1] = 0;
            padding_params[2] = wpad - wpad / 2;
            padding_params[3] = wpad / 2;
            padding_params[4] = 0;
            padding_params[5] = 0;

            std::vector<VkMat> padding_inputs(2);
            padding_inputs[0] = bottom_blob;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkMat> padding_outputs(1);
            padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);
            bottom_blob_bordered = padding_outputs[0];
        }
    }

    w = bottom_blob_bordered.w;

    int outw = (w - kernel_extent_w) / stride_w + 1;

    int out_elempack = num_output % 4 == 0 ? 4 : 1;

    size_t out_elemsize = elemsize / elempack * out_elempack;

    const int maxk = kernel_w;
    const int num_input = bottom_blob_bordered.h * elempack;

    bool is_conv1x1s1d1 = kernel_w == 1 && stride_w == 1 && dilation_w == 1;

    bool use_gemm = opt.use_sgemm_convolution
                    && !is_conv1x1s1d1
                    && num_input * maxk >= 8
                    && num_output >= 8;

    if (use_radish_gemm)
    {
        if (!pipeline_convolution1d_gemm || bottom_blob_bordered.dims != 2 || outw <= 0
            || bottom_blob_bordered.h * elempack != num_input)
            return -1;
        top_blob.create(outw, num_output / 4, (size_t)16u, 4, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
        std::vector<VkMat> bindings(4);
        bindings[0] = weight_data_gpu;
        bindings[1] = bottom_blob_bordered;
        bindings[2] = top_blob;
        bindings[3] = bias_term ? bias_data_gpu : weight_data_gpu;
        std::vector<vk_constant_type> constants(4);
        constants[0].i = num_output;
        constants[1].i = outw;
        constants[2].i = bottom_blob_bordered.w;
        constants[3].i = outw;
        VkMat dispatcher;
        dispatcher.w = (outw + 63) / 64 * 16;
        dispatcher.h = (num_output + 63) / 64 * 8;
        dispatcher.c = 1;
        cmd.record_pipeline(pipeline_convolution1d_gemm, bindings, constants, dispatcher);
        return 0;
    }

    if (use_radish_tiled)
    {
        if (!pipeline_convolution1d_gemm || pipeline_convolution1d_gemm->pipeline() == VK_NULL_HANDLE
            || bottom_blob_bordered.dims != 2 || elempack != 4 || elemsize != 16u
            || num_input != weight_data_size / maxk / num_output || outw <= 0)
            return -1;

        top_blob.create(outw, num_output / 4, (size_t)16u, 4, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        std::vector<VkMat> bindings(4);
        bindings[0] = weight_data_gpu;
        bindings[1] = bottom_blob_bordered;
        bindings[2] = top_blob;
        bindings[3] = bias_data_gpu;

        std::vector<vk_constant_type> constants(2);
        constants[0].i = bottom_blob_bordered.w;
        constants[1].i = outw;

        VkMat dispatcher;
        const int tile_width = radish_tiled32 ? 32 : 64;
        const int tile_height = radish_tiled32 ? 32 : 64;
        dispatcher.w = (outw / tile_width + (outw % tile_width != 0)) * (radish_tiled32 ? 8 : 16);
        dispatcher.h = (num_output / tile_height) * (radish_tiled32 ? 4 : 8);
        dispatcher.c = 1;
        cmd.record_pipeline(pipeline_convolution1d_gemm, bindings, constants, dispatcher);
        return 0;
    }

    if (use_gemm && pipeline_convolution1d_gemm)
    {
        if (use_cooperative_matrix)
        {
            top_blob.create(outw, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
            if (top_blob.empty())
                return -100;

            std::vector<VkMat> bindings(6);
            bindings[0] = bottom_blob_bordered;
            bindings[1] = top_blob;
            bindings[2] = weight_data_gpu;
            bindings[3] = bias_data_gpu;
            bindings[4] = bottom_blob_bordered;
            bindings[5] = top_blob;

            std::vector<vk_constant_type> constants(4);
            constants[0].u32 = bottom_blob_bordered.w;
            constants[1].u32 = bottom_blob_bordered.w;
            constants[2].u32 = top_blob.w;
            constants[3].u32 = top_blob.w;

            const int blocks_x = (top_blob.w + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
            const int blocks_y = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

            VkMat dispatcher;
            dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
            dispatcher.h = 1;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_convolution1d_gemm, bindings, constants, dispatcher);

            return 0;
        }
        else
        {
            const int num_input_packed = (num_input + 3) / 4 * 4;
            const int num_output_packed = (num_output + 3) / 4 * 4;

            top_blob.create(outw, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
            if (top_blob.empty())
                return -100;

            std::vector<VkMat> bindings(6);
            bindings[0] = bottom_blob_bordered;
            bindings[1] = top_blob;
            bindings[2] = bottom_blob_bordered;
            bindings[3] = top_blob;
            bindings[4] = weight_data_gpu;
            bindings[5] = bias_data_gpu;

            std::vector<vk_constant_type> constants(6);
            constants[0].i = bottom_blob_bordered.w;
            constants[1].i = num_input_packed / 4;
            constants[2].i = top_blob.w;
            constants[3].i = num_output_packed / 4;
            constants[4].i = num_output;
            constants[5].i = num_input;

            VkMat dispatcher;
            dispatcher.w = (top_blob.w + 3) / 4;
            dispatcher.h = num_output_packed / 4;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_convolution1d_gemm, bindings, constants, dispatcher);

            return 0;
        }
    }

    if (is_conv1x1s1d1 && pipeline_convolution1d_1x1s1d1)
    {
        if (use_cooperative_matrix)
        {
            top_blob.create(outw, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
            if (top_blob.empty())
                return -100;

            std::vector<VkMat> bindings(6);
            bindings[0] = bottom_blob_bordered;
            bindings[1] = top_blob;
            bindings[2] = weight_data_gpu;
            bindings[3] = bias_data_gpu;
            bindings[4] = bottom_blob_bordered;
            bindings[5] = top_blob;

            std::vector<vk_constant_type> constants(3);
            constants[0].u32 = bottom_blob_bordered.w;
            constants[1].u32 = bottom_blob_bordered.w;
            constants[2].u32 = top_blob.w;

            const int blocks_x = (top_blob.w + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
            const int blocks_y = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

            VkMat dispatcher;
            dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
            dispatcher.h = 1;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_convolution1d_1x1s1d1, bindings, constants, dispatcher);

            return 0;
        }
        else
        {
            const int num_input_packed = (num_input + 3) / 4 * 4;
            const int num_output_packed = (num_output + 3) / 4 * 4;

            top_blob.create(outw, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
            if (top_blob.empty())
                return -100;

            const int outh_pack4 = num_output_packed / 4;

            std::vector<VkMat> bindings(6);
            bindings[0] = bottom_blob_bordered;
            bindings[1] = top_blob;
            bindings[2] = bottom_blob_bordered;
            bindings[3] = top_blob;
            bindings[4] = weight_data_gpu;
            bindings[5] = bias_data_gpu;

            std::vector<vk_constant_type> constants(6);
            constants[0].i = bottom_blob_bordered.w;
            constants[1].i = bottom_blob_bordered.h;
            constants[2].i = top_blob.w;
            constants[3].i = outh_pack4;
            constants[4].i = num_output;
            constants[5].i = num_input_packed;

            VkMat dispatcher;
            dispatcher.w = (top_blob.w + 1) / 2;
            dispatcher.h = (outh_pack4 + 1) / 2;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_convolution1d_1x1s1d1, bindings, constants, dispatcher);

            return 0;
        }
    }

    const int num_output_packed = (num_output + 3) / 4 * 4;

    top_blob.create(outw, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    const int outh_pack4 = num_output_packed / 4;

    std::vector<VkMat> bindings(6);
    bindings[0] = bottom_blob_bordered;
    bindings[1] = top_blob;
    bindings[2] = bottom_blob_bordered;
    bindings[3] = top_blob;
    bindings[4] = weight_data_gpu;
    bindings[5] = bias_data_gpu;

    std::vector<vk_constant_type> constants(5);
    constants[0].i = bottom_blob_bordered.w;
    constants[1].i = bottom_blob_bordered.h;
    constants[2].i = top_blob.w;
    constants[3].i = outh_pack4;
    constants[4].i = num_output;

    VkMat dispatcher;
    dispatcher.w = (top_blob.w + 1) / 2;
    dispatcher.h = (outh_pack4 + 1) / 2;
    dispatcher.c = 1;

    cmd.record_pipeline(pipeline_convolution1d, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
