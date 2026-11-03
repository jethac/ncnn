// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "instancenorm_vulkan.h"

#include <limits.h>
#include <limits>
#include <stdlib.h>
#include <string.h>

#include "layer_shader_type.h"

namespace ncnn {

static bool radish_norm_sum4_fp32(const Option& opt)
{
    return opt.use_vulkan_compute && opt.use_packing_layout
           && !opt.use_fp16_packed && !opt.use_fp16_storage
           && !opt.use_fp16_arithmetic && !opt.use_fp16_uniform
           && !opt.use_bf16_packed && !opt.use_bf16_storage
           && !opt.use_int8_inference && !opt.use_int8_packed && !opt.use_int8_storage
           && !opt.use_int8_arithmetic && !opt.use_int8_uniform
           && !opt.use_int16_packed && !opt.use_int16_storage
           && !opt.use_subgroup_ops && !opt.use_shader_local_memory
           && !opt.use_cooperative_matrix && !opt.use_tensor_storage;
}

#if NCNN_STRING
static bool radish_norm_outplace_name(const std::string& name, const char* suffix)
{
    const char* p = name.c_str();
    if (strncmp(p, "radish_norm_", 12) != 0)
        return false;
    p += 12;
    if (*p < '0' || *p > '9')
        return false;
    while (*p >= '0' && *p <= '9')
        p++;
    return strcmp(p, suffix) == 0;
}
#endif

static bool radish_norm_outplace_buffer(const VkMat& blob, int width, int packs, const VulkanDevice* vkdev, const VkAllocator* allocator)
{
    if (!allocator || allocator->vkdev != vkdev || blob.allocator != allocator
        || !blob.data || !blob.refcount || *blob.refcount <= 0 || !blob.data->buffer
        || blob.dims != 3 || blob.n != 1 || blob.h != 1 || blob.d != 1
        || blob.w != width || blob.c != packs || width <= 0 || packs <= 0
        || blob.elempack != 4 || blob.elemsize != 16u || blob.cstep != (size_t)width)
        return false;
    const size_t max_size = std::numeric_limits<size_t>::max();
    if ((size_t)width > max_size / (size_t)packs)
        return false;
    const size_t count = (size_t)width * packs;
    if (count > max_size / 16u || blob.total() != count
#if NCNN_BATCH
        || blob.nstep != count
#endif
        || blob.offset > blob.data->capacity || count * 16u > blob.data->capacity - blob.offset
        || blob.buffer_capacity() != count * 16u || blob.data->offset > max_size - blob.offset)
        return false;
    const size_t offset = blob.buffer_offset();
    const size_t alignment = vkdev->info.physicalDeviceProperties().limits.minStorageBufferOffsetAlignment;
    return offset % 16u == 0 && (!alignment || offset % alignment == 0)
           && count * 16u <= max_size - offset
           && count * 16u <= vkdev->info.physicalDeviceProperties().limits.maxStorageBufferRange;
}

InstanceNorm_vulkan::InstanceNorm_vulkan()
{
    support_vulkan = true;
    radish_norm_outplace = false;
    pipeline_instancenorm_radish_outplace_pack4 = 0;
    support_vulkan_packing = true;

    pipeline_instancenorm_reduce_sum4_fp16_to_fp32 = 0;
    pipeline_instancenorm_reduce_sum4_fp32[0] = 0;
    pipeline_instancenorm_reduce_sum4_fp32[1] = 0;
    pipeline_instancenorm_reduce_mean = 0;
    pipeline_instancenorm_sub_mean_square = 0;
    pipeline_instancenorm_coeffs = 0;
    pipeline_instancenorm_norm = 0;

    pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack4 = 0;
    pipeline_instancenorm_reduce_sum4_fp32_pack4[0] = 0;
    pipeline_instancenorm_reduce_sum4_fp32_pack4[1] = 0;
    pipeline_instancenorm_reduce_mean_pack4 = 0;
    pipeline_instancenorm_sub_mean_square_pack4 = 0;
    pipeline_instancenorm_radish_sum4_pack4 = 0;
    pipeline_instancenorm_coeffs_pack4 = 0;
    pipeline_instancenorm_norm_pack4 = 0;

    pipeline_instancenorm_reduce_subgroup = 0;
    pipeline_instancenorm_reduce_subgroup_pack4 = 0;
}

int InstanceNorm_vulkan::create_pipeline(const Option& opt)
{
    const int subgroup_size = vkdev->info.subgroup_size();
    const uint32_t support_subgroup_ops = vkdev->info.support_subgroup_ops();
    const uint32_t required_subgroup_ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
    bool use_subgroup_ops = opt.use_subgroup_ops && ((support_subgroup_ops & required_subgroup_ops) == required_subgroup_ops);
    if (subgroup_size < 4 || subgroup_size > 128)
    {
        // sanitize wired subgroup_size
        use_subgroup_ops = false;
    }
    const Mat& shape = top_shapes.empty() ? Mat() : top_shapes[0];

    const char* norm_sum4 = getenv("RADISH_NCNN_NORM_SUM4");
    const bool use_radish_norm_sum4 = norm_sum4 && strcmp(norm_sum4, "1") == 0
                                     && vkdev->info.vendor_id() == 0x5143
                                     && vkdev->info.device_id() == 506
                                     && strcmp(vkdev->info.device_name(), "Radish Adreno 506 (experimental compute)") == 0
                                     && radish_norm_sum4_fp32(opt) && !use_subgroup_ops
                                     && affine == 0 && eps == 1.e-5f
                                     && (channels == 128 || channels == 256 || channels == 512 || channels == 1024)
                                     && shape.dims == 3 && shape.n == 1 && shape.h == 1 && shape.d == 1
                                     && shape.elempack == 4 && shape.elemsize == 16u && shape.c == channels / 4
                                     && shape.w > 0 && shape.w <= INT_MAX - 3;

    bool use_radish_norm_outplace = false;
#if NCNN_STRING
    const char* norm_outplace = getenv("RADISH_NCNN_NORM_OUTPLACE");
    const char* reshape_alias = getenv("RADISH_NCNN_RESHAPE_ALIAS");
    if (use_radish_norm_sum4 && norm_outplace && strcmp(norm_outplace, "1") == 0
        && reshape_alias && strcmp(reshape_alias, "1") == 0
        && radish_norm_outplace_name(name, "_instance") && bottom_shapes.size() == 1)
    {
        const Mat& input_shape = bottom_shapes[0];
        use_radish_norm_outplace = input_shape.dims == 3 && input_shape.n == 1
                                  && input_shape.w == shape.w && input_shape.h == 1 && input_shape.d == 1
                                  && input_shape.c == shape.c && input_shape.elempack == 4 && input_shape.elemsize == 16u
                                  && input_shape.cstep == (size_t)shape.w && shape.cstep == (size_t)shape.w
                                  && (size_t)shape.w <= vkdev->info.physicalDeviceProperties().limits.maxStorageBufferRange / 16u / shape.c;
    }
#endif

    int _channels = channels;
    if (shape.dims == 3 || shape.dims == 4) _channels = shape.c * shape.elempack;

    int elempack = 1;
    if (_channels != 0) elempack = _channels % 4 == 0 ? 4 : 1;

    size_t elemsize;
    if (opt.use_fp16_storage || opt.use_fp16_packed || opt.use_bf16_storage || opt.use_bf16_packed)
    {
        elemsize = elempack * 2u;
    }
    else
    {
        elemsize = elempack * 4u;
    }

    // TODO resolve workspace_shape.w
    Mat workspace_shape;
    if (_channels != 0) workspace_shape = Mat(1, 1, _channels / elempack, (void*)0, elemsize, elempack);

    int shape_dims = std::min(3, shape.dims);
    int shape_w = shape.w;
    int shape_h = shape.h * shape.d;
    int shape_c = shape.c * shape.elempack / elempack;
    int shape_cstep = shape.cstep;
    if (shape.dims == 0)
    {
        shape_dims = 0;
        shape_w = 0;
        shape_h = 0;
        shape_c = 0;
        shape_cstep = 0;
    }

    if (use_subgroup_ops)
    {
        pipeline_instancenorm_reduce_subgroup = new Pipeline(vkdev);
        pipeline_instancenorm_reduce_subgroup->set_subgroup_size(subgroup_size);
        pipeline_instancenorm_reduce_subgroup->set_local_size_xyz(subgroup_size, 1, 1);
        pipeline_instancenorm_reduce_subgroup->create(LayerShaderType::instancenorm_reduce_subgroup, opt, std::vector<vk_specialization_type>());

        pipeline_instancenorm_reduce_subgroup_pack4 = new Pipeline(vkdev);
        pipeline_instancenorm_reduce_subgroup_pack4->set_subgroup_size(subgroup_size);
        pipeline_instancenorm_reduce_subgroup_pack4->set_local_size_xyz(subgroup_size, 1, 1);
        pipeline_instancenorm_reduce_subgroup_pack4->create(LayerShaderType::instancenorm_reduce_subgroup_pack4, opt, std::vector<vk_specialization_type>());
    }
    else
    {
        {
            Mat local_size_xyz;
            {
                local_size_xyz = Mat(16, 1, _channels ? std::min(4, _channels / elempack) : 4, (void*)0);
                if (workspace_shape.dims != 0)
                {
                    local_size_xyz.w = 16;
                    local_size_xyz.h = 1;
                    local_size_xyz.c = std::min(4, workspace_shape.c);
                }
            }

            // pack1
            if (elempack == 1 || _channels == 0)
            {
                pipeline_instancenorm_reduce_sum4_fp16_to_fp32 = new Pipeline(vkdev);
                pipeline_instancenorm_reduce_sum4_fp16_to_fp32->set_optimal_local_size_xyz(local_size_xyz);
                pipeline_instancenorm_reduce_sum4_fp16_to_fp32->create(LayerShaderType::instancenorm_reduce_sum4_fp16_to_fp32, opt, std::vector<vk_specialization_type>());

                pipeline_instancenorm_reduce_sum4_fp32[0] = new Pipeline(vkdev);
                pipeline_instancenorm_reduce_sum4_fp32[0]->set_optimal_local_size_xyz(local_size_xyz);
                pipeline_instancenorm_reduce_sum4_fp32[0]->create(LayerShaderType::instancenorm_reduce_sum4_fp32, opt, std::vector<vk_specialization_type>());
                pipeline_instancenorm_reduce_sum4_fp32[1] = new Pipeline(vkdev);
                pipeline_instancenorm_reduce_sum4_fp32[1]->set_optimal_local_size_xyz(local_size_xyz);
                pipeline_instancenorm_reduce_sum4_fp32[1]->create(LayerShaderType::instancenorm_reduce_sum4_fp32, opt, std::vector<vk_specialization_type>());
            }

            // pack4
            if (elempack == 4 || _channels == 0)
            {
                pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack4 = new Pipeline(vkdev);
                pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack4->set_optimal_local_size_xyz(local_size_xyz);
                pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack4->create(LayerShaderType::instancenorm_reduce_sum4_fp16_to_fp32_pack4, opt, std::vector<vk_specialization_type>());

                pipeline_instancenorm_reduce_sum4_fp32_pack4[0] = new Pipeline(vkdev);
                pipeline_instancenorm_reduce_sum4_fp32_pack4[0]->set_optimal_local_size_xyz(local_size_xyz);
                pipeline_instancenorm_reduce_sum4_fp32_pack4[0]->create(LayerShaderType::instancenorm_reduce_sum4_fp32_pack4, opt, std::vector<vk_specialization_type>());
                pipeline_instancenorm_reduce_sum4_fp32_pack4[1] = new Pipeline(vkdev);
                pipeline_instancenorm_reduce_sum4_fp32_pack4[1]->set_optimal_local_size_xyz(local_size_xyz);
                pipeline_instancenorm_reduce_sum4_fp32_pack4[1]->create(LayerShaderType::instancenorm_reduce_sum4_fp32_pack4, opt, std::vector<vk_specialization_type>());

                if (use_radish_norm_sum4)
                {
                    pipeline_instancenorm_radish_sum4_pack4 = new Pipeline(vkdev);
                    pipeline_instancenorm_radish_sum4_pack4->set_local_size_xyz(16, 1, 4);
                    int ret = pipeline_instancenorm_radish_sum4_pack4->create(LayerShaderType::instancenorm_radish_sum4_pack4, opt, std::vector<vk_specialization_type>());
                    if (ret != 0)
                        return ret;
                    if (!pipeline_instancenorm_radish_sum4_pack4->pipeline()
                        || pipeline_instancenorm_radish_sum4_pack4->local_size_x() != 16
                        || pipeline_instancenorm_radish_sum4_pack4->local_size_y() != 1
                        || pipeline_instancenorm_radish_sum4_pack4->local_size_z() != 4
                        || pipeline_instancenorm_radish_sum4_pack4->shader_info().binding_count != 3
                        || pipeline_instancenorm_radish_sum4_pack4->shader_info().push_constant_count != 8)
                        return -1;
                }
            }
        }

        {
            std::vector<vk_specialization_type> specializations(0 + 4);
            specializations[0].i = 0; // TODO resolve workspace_shape.w;
            specializations[1].i = 0; // TODO resolve workspace_shape.h;
            specializations[2].i = workspace_shape.c;
            specializations[3].i = 0; // TODO resolve workspace_shape.cstep;

            Mat local_size_xyz(_channels ? std::min(64, _channels / elempack) : 64, 1, 1, (void*)0);
            if (workspace_shape.dims != 0)
            {
                local_size_xyz.w = std::min(64, workspace_shape.c);
                local_size_xyz.h = 1;
                local_size_xyz.c = 1;
            }

            if (elempack == 1 || _channels == 0)
            {
                pipeline_instancenorm_reduce_mean = new Pipeline(vkdev);
                pipeline_instancenorm_reduce_mean->set_optimal_local_size_xyz(local_size_xyz);
                pipeline_instancenorm_reduce_mean->create(LayerShaderType::instancenorm_reduce_mean, opt, specializations);
            }

            if (elempack == 4 || _channels == 0)
            {
                pipeline_instancenorm_reduce_mean_pack4 = new Pipeline(vkdev);
                pipeline_instancenorm_reduce_mean_pack4->set_optimal_local_size_xyz(local_size_xyz);
                pipeline_instancenorm_reduce_mean_pack4->create(LayerShaderType::instancenorm_reduce_mean_pack4, opt, specializations);
            }
        }

        Mat square_workspace;
        if (shape.dims == 3 || shape.dims == 4) square_workspace = Mat(shape.w, shape.h * shape.d, shape.c * shape.elempack / elempack, (void*)0, elempack * 4u, elempack);

        {
            std::vector<vk_specialization_type> specializations(0 + 10);
            specializations[0 + 0].i = shape_dims;
            specializations[0 + 1].i = shape_w;
            specializations[0 + 2].i = shape_h;
            specializations[0 + 3].i = shape_c;
            specializations[0 + 4].i = shape_cstep;
            specializations[0 + 5].i = square_workspace.dims;
            specializations[0 + 6].i = square_workspace.w;
            specializations[0 + 7].i = square_workspace.h;
            specializations[0 + 8].i = square_workspace.c;
            specializations[0 + 9].i = square_workspace.cstep;

            Mat local_size_xyz(4, 4, _channels ? std::min(4, _channels / elempack) : 4, (void*)0);
            if (square_workspace.dims != 0)
            {
                local_size_xyz.w = std::min(4, square_workspace.w);
                local_size_xyz.h = std::min(4, square_workspace.h);
                local_size_xyz.c = std::min(4, square_workspace.c);
            }

            if (elempack == 1 || _channels == 0)
            {
                pipeline_instancenorm_sub_mean_square = new Pipeline(vkdev);
                pipeline_instancenorm_sub_mean_square->set_optimal_local_size_xyz(local_size_xyz);
                pipeline_instancenorm_sub_mean_square->create(LayerShaderType::instancenorm_sub_mean_square, opt, specializations);
            }

            if (elempack == 4 || _channels == 0)
            {
                pipeline_instancenorm_sub_mean_square_pack4 = new Pipeline(vkdev);
                pipeline_instancenorm_sub_mean_square_pack4->set_optimal_local_size_xyz(local_size_xyz);
                pipeline_instancenorm_sub_mean_square_pack4->create(LayerShaderType::instancenorm_sub_mean_square_pack4, opt, specializations);
            }
        }
    }

    {
        std::vector<vk_specialization_type> specializations(3);
        specializations[0].f = eps;
        specializations[1].i = affine;
        specializations[2].i = _channels / elempack;

        Mat local_size_xyz(_channels ? std::min(64, _channels / elempack) : 64, 1, 1, (void*)0);
        if (workspace_shape.dims != 0)
        {
            local_size_xyz.w = std::min(64, workspace_shape.c);
            local_size_xyz.h = 1;
            local_size_xyz.c = 1;
        }

        if (elempack == 1 || _channels == 0)
        {
            pipeline_instancenorm_coeffs = new Pipeline(vkdev);
            pipeline_instancenorm_coeffs->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_coeffs->create(LayerShaderType::instancenorm_coeffs, opt, specializations);
        }

        if (elempack == 4 || _channels == 0)
        {
            pipeline_instancenorm_coeffs_pack4 = new Pipeline(vkdev);
            pipeline_instancenorm_coeffs_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_coeffs_pack4->create(LayerShaderType::instancenorm_coeffs_pack4, opt, specializations);
        }
    }

    {
        std::vector<vk_specialization_type> specializations(0 + 5);
        specializations[0 + 0].i = shape_dims;
        specializations[0 + 1].i = shape_w;
        specializations[0 + 2].i = shape_h;
        specializations[0 + 3].i = shape_c;
        specializations[0 + 4].i = shape_cstep;

        Mat local_size_xyz(4, 4, _channels ? std::min(4, _channels / elempack) : 4, (void*)0);
        if (shape_dims != 0)
        {
            local_size_xyz.w = std::min(4, shape_w);
            local_size_xyz.h = std::min(4, shape_h);
            local_size_xyz.c = std::min(4, shape_c);
        }

        if (elempack == 1 || _channels == 0)
        {
            pipeline_instancenorm_norm = new Pipeline(vkdev);
            pipeline_instancenorm_norm->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_norm->create(LayerShaderType::instancenorm_norm, opt, specializations);
        }

        if (elempack == 4 || _channels == 0)
        {
            pipeline_instancenorm_norm_pack4 = new Pipeline(vkdev);
            pipeline_instancenorm_norm_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_norm_pack4->create(LayerShaderType::instancenorm_norm_pack4, opt, specializations);

            if (use_radish_norm_outplace)
            {
                pipeline_instancenorm_radish_outplace_pack4 = new Pipeline(vkdev);
                pipeline_instancenorm_radish_outplace_pack4->set_optimal_local_size_xyz(local_size_xyz);
                int ret = pipeline_instancenorm_radish_outplace_pack4->create(LayerShaderType::instancenorm_radish_outplace_pack4, opt, specializations);
                if (ret != 0)
                    return ret;
                const Pipeline* pipeline = pipeline_instancenorm_radish_outplace_pack4;
                if (!pipeline->pipeline() || !pipeline_instancenorm_norm_pack4->pipeline()
                    || !pipeline_instancenorm_radish_sum4_pack4 || !pipeline_instancenorm_radish_sum4_pack4->pipeline()
                    || !pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack4->pipeline()
                    || !pipeline_instancenorm_reduce_sum4_fp32_pack4[0]->pipeline()
                    || !pipeline_instancenorm_reduce_sum4_fp32_pack4[1]->pipeline()
                    || !pipeline_instancenorm_reduce_mean_pack4->pipeline()
                    || !pipeline_instancenorm_coeffs_pack4->pipeline()
                    || pipeline->local_size_x() != pipeline_instancenorm_norm_pack4->local_size_x()
                    || pipeline->local_size_y() != pipeline_instancenorm_norm_pack4->local_size_y()
                    || pipeline->local_size_z() != pipeline_instancenorm_norm_pack4->local_size_z()
                    || pipeline->shader_info().binding_count != 3 || pipeline->shader_info().push_constant_count != 5
                    || pipeline->shader_info().specialization_count != 5)
                    return -1;
                for (int i = 0; i < 3; i++)
                    if (pipeline->shader_info().binding_types[i] != 1)
                        return -1;
                radish_norm_outplace = true;
                support_inplace = false;
            }
        }
    }

    return 0;
}

int InstanceNorm_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_instancenorm_radish_outplace_pack4;
    pipeline_instancenorm_radish_outplace_pack4 = 0;
    radish_norm_outplace = false;
    support_inplace = true;

    delete pipeline_instancenorm_reduce_sum4_fp16_to_fp32;
    pipeline_instancenorm_reduce_sum4_fp16_to_fp32 = 0;

    delete pipeline_instancenorm_reduce_sum4_fp32[0];
    delete pipeline_instancenorm_reduce_sum4_fp32[1];
    pipeline_instancenorm_reduce_sum4_fp32[0] = 0;
    pipeline_instancenorm_reduce_sum4_fp32[1] = 0;

    delete pipeline_instancenorm_reduce_mean;
    pipeline_instancenorm_reduce_mean = 0;

    delete pipeline_instancenorm_sub_mean_square;
    pipeline_instancenorm_sub_mean_square = 0;

    delete pipeline_instancenorm_coeffs;
    pipeline_instancenorm_coeffs = 0;

    delete pipeline_instancenorm_norm;
    pipeline_instancenorm_norm = 0;

    delete pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack4;
    pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack4 = 0;

    delete pipeline_instancenorm_reduce_sum4_fp32_pack4[0];
    delete pipeline_instancenorm_reduce_sum4_fp32_pack4[1];
    pipeline_instancenorm_reduce_sum4_fp32_pack4[0] = 0;
    pipeline_instancenorm_reduce_sum4_fp32_pack4[1] = 0;

    delete pipeline_instancenorm_reduce_mean_pack4;
    pipeline_instancenorm_reduce_mean_pack4 = 0;

    delete pipeline_instancenorm_sub_mean_square_pack4;
    pipeline_instancenorm_sub_mean_square_pack4 = 0;

    delete pipeline_instancenorm_radish_sum4_pack4;
    pipeline_instancenorm_radish_sum4_pack4 = 0;

    delete pipeline_instancenorm_coeffs_pack4;
    pipeline_instancenorm_coeffs_pack4 = 0;

    delete pipeline_instancenorm_norm_pack4;
    pipeline_instancenorm_norm_pack4 = 0;

    delete pipeline_instancenorm_reduce_subgroup;
    pipeline_instancenorm_reduce_subgroup = 0;
    delete pipeline_instancenorm_reduce_subgroup_pack4;
    pipeline_instancenorm_reduce_subgroup_pack4 = 0;

    return 0;
}

int InstanceNorm_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (affine == 0)
        return 0;

    cmd.record_upload(gamma_data, gamma_data_gpu, opt);

    cmd.record_upload(beta_data, beta_data_gpu, opt);

    return 0;
}

int InstanceNorm_vulkan::forward_radish_coefficients(const VkMat& input, VkMat& coefficients, VkCompute& cmd, const Option& opt) const
{
    if (&input == &coefficients || coefficients.data || coefficients.refcount || !coefficients.empty()
        || !vkdev || !vkdev->is_valid() || vkdev->info.vendor_id() != 0x5143 || vkdev->info.device_id() != 506
        || strcmp(vkdev->info.device_name(), "Radish Adreno 506 (experimental compute)") != 0
        || !radish_norm_sum4_fp32(opt) || affine != 0 || eps != 1.e-5f
        || (channels != 128 && channels != 256)
        || !opt.workspace_vkallocator || opt.workspace_vkallocator->vkdev != vkdev
        || bottom_shapes.size() != 1 || top_shapes.size() != 1
        || input.w <= 0 || input.w > INT_MAX - 3
        || !radish_norm_outplace_buffer(input, input.w, channels / 4, vkdev, opt.blob_vkallocator)
        || !input.data->memory
        || pipeline_instancenorm_reduce_subgroup || pipeline_instancenorm_reduce_subgroup_pack4)
        return -1;
    for (int i = 0; i < 2; i++)
    {
        const Mat& shape = i == 0 ? bottom_shapes[0] : top_shapes[0];
        if (shape.dims != 3 || shape.n != 1 || shape.w != input.w || shape.h != 1 || shape.d != 1
            || shape.c != channels / 4 || shape.elempack != 4 || shape.elemsize != 16u || shape.cstep != (size_t)input.w
            || shape.total() != input.total()
#if NCNN_BATCH
            || shape.nstep != input.nstep
#endif
        )
            return -1;
    }
    const Pipeline* pipelines[] = {pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack4,
                                   pipeline_instancenorm_reduce_sum4_fp32_pack4[0],
                                   pipeline_instancenorm_reduce_sum4_fp32_pack4[1],
                                   pipeline_instancenorm_reduce_mean_pack4,
                                   pipeline_instancenorm_radish_sum4_pack4,
                                   pipeline_instancenorm_coeffs_pack4};
    const int bindings[] = {2, 2, 2, 2, 3, 5};
    const int pushes[] = {8, 8, 8, 5, 8, 1};
    for (int i = 0; i < 6; i++)
    {
        const Pipeline* pipeline = pipelines[i];
        if (!pipeline || !pipeline->pipeline() || pipeline->shader_info().binding_count != bindings[i]
            || pipeline->shader_info().push_constant_count != pushes[i])
            return -1;
        for (int j = 0; j < bindings[i]; j++)
            if (pipeline->shader_info().binding_types[j] != 1)
                return -1;
    }
    VkMat unused_output;
    return forward_impl(input, unused_output, cmd, opt, false, &coefficients);
}

int InstanceNorm_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    if (!radish_norm_outplace)
        return InstanceNorm::forward(bottom_blob, top_blob, cmd, opt);
    if (&bottom_blob == &top_blob || !radish_norm_sum4_fp32(opt)
        || !pipeline_instancenorm_radish_outplace_pack4 || !pipeline_instancenorm_radish_sum4_pack4
        || top_shapes.size() != 1 || bottom_shapes.size() != 1
        || bottom_blob.w <= 0 || bottom_blob.w > INT_MAX - 3 || bottom_blob.w != top_shapes[0].w
        || bottom_blob.w != bottom_shapes[0].w || bottom_blob.c != channels / 4
        || !radish_norm_outplace_buffer(bottom_blob, bottom_blob.w, channels / 4, vkdev, opt.blob_vkallocator))
        return -1;

    VkMat output(bottom_blob.w, 1, bottom_blob.c, 16u, 4, opt.blob_vkallocator);
    if (output.empty())
        return -100;
    if (!radish_norm_outplace_buffer(output, bottom_blob.w, bottom_blob.c, vkdev, opt.blob_vkallocator)
        || output.data == bottom_blob.data)
        return -1;
    if (output.buffer() == bottom_blob.buffer()
        && output.buffer_offset() < bottom_blob.buffer_offset() + bottom_blob.buffer_capacity()
        && bottom_blob.buffer_offset() < output.buffer_offset() + output.buffer_capacity())
        return -1;

    int ret = forward_impl(bottom_blob, output, cmd, opt, true);
    if (ret != 0)
        return ret;
    top_blob = output;
    return 0;
}

int InstanceNorm_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    return forward_impl(bottom_top_blob, bottom_top_blob, cmd, opt, false);
}

int InstanceNorm_vulkan::forward_impl(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt, bool outplace, VkMat* coefficients_out) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h * bottom_blob.d;
    int c = bottom_blob.c;
    int size = w * h;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const bool use_radish_norm_sum4 = pipeline_instancenorm_radish_sum4_pack4
                                     && radish_norm_sum4_fp32(opt)
                                     && bottom_blob.dims == 3 && bottom_blob.n == 1
                                     && bottom_blob.h == 1 && bottom_blob.d == 1
                                     && elempack == 4 && elemsize == 16u && c > 0 && c == channels / 4
                                     && !top_shapes.empty() && w == top_shapes[0].w
                                     && w > 0 && w <= INT_MAX - 3 && bottom_blob.cstep == (size_t)w
                                     && (size_t)w <= vkdev->info.physicalDeviceProperties().limits.maxStorageBufferRange / 16u / c;

    // mean and var
    VkMat mean_workspace(c, 4u * elempack, elempack, opt.workspace_vkallocator);
    VkMat var_workspace(c, 4u * elempack, elempack, opt.workspace_vkallocator);
    if (use_radish_norm_sum4 && (mean_workspace.empty() || var_workspace.empty()))
        return -100;

    const Pipeline* pipeline_reduce_subgroup = elempack == 4 ? pipeline_instancenorm_reduce_subgroup_pack4 : pipeline_instancenorm_reduce_subgroup;
    if (pipeline_reduce_subgroup)
    {
        std::vector<VkMat> bindings(3);
        bindings[0] = bottom_blob;
        bindings[1] = mean_workspace;
        bindings[2] = var_workspace;

        std::vector<vk_constant_type> constants(3);
        constants[0].i = bottom_blob.cstep;
        constants[1].i = size;
        constants[2].i = c;

        // split channels across x and y to stay within the dispatch limits
        const int group_count_x = (c - 1) / vkdev->info.max_workgroup_count_y() + 1;
        const int group_count_y = (c - 1) / group_count_x + 1;

        VkMat dispatcher;
        dispatcher.w = group_count_x * pipeline_reduce_subgroup->local_size_x();
        dispatcher.h = group_count_y;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_reduce_subgroup, bindings, constants, dispatcher);
    }
    else
    {
        // mean
        {
            // reduce sum
            VkMat sum_workspace;
            {
                int reduced_w = (size + 3) / 4;
                int reduced_h = 1;
                int reduced_c = bottom_blob.c;

                sum_workspace.create(reduced_w, reduced_h, reduced_c, 4u * elempack, elempack, opt.workspace_vkallocator);
                if (coefficients_out && sum_workspace.empty())
                    return -100;
                {
                    std::vector<VkMat> bindings(2);
                    bindings[0] = bottom_blob;
                    bindings[1] = sum_workspace;

                    std::vector<vk_constant_type> constants(8);
                    constants[0].i = size;
                    constants[1].i = 1;
                    constants[2].i = bottom_blob.c;
                    constants[3].i = bottom_blob.cstep;
                    constants[4].i = sum_workspace.w;
                    constants[5].i = 1;
                    constants[6].i = sum_workspace.c;
                    constants[7].i = sum_workspace.cstep;

                    const Pipeline* pipeline = elempack == 4 ? pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack4 : pipeline_instancenorm_reduce_sum4_fp16_to_fp32;

                    cmd.record_pipeline(pipeline, bindings, constants, sum_workspace);
                }
            }

            int pb = 0;
            while (sum_workspace.w > 4)
            {
                int reduced_w = (sum_workspace.w + 3) / 4;
                int reduced_h = 1;
                int reduced_c = sum_workspace.c;

                VkMat sum_workspace_reduced;
                sum_workspace_reduced.create(reduced_w, reduced_h, reduced_c, 4u * elempack, elempack, opt.workspace_vkallocator);
                if (coefficients_out && sum_workspace_reduced.empty())
                    return -100;

                {
                    std::vector<VkMat> bindings(2);
                    bindings[0] = sum_workspace;
                    bindings[1] = sum_workspace_reduced;

                    std::vector<vk_constant_type> constants(8);
                    constants[0].i = sum_workspace.w;
                    constants[1].i = 1;
                    constants[2].i = sum_workspace.c;
                    constants[3].i = sum_workspace.cstep;
                    constants[4].i = sum_workspace_reduced.w;
                    constants[5].i = 1;
                    constants[6].i = sum_workspace_reduced.c;
                    constants[7].i = sum_workspace_reduced.cstep;

                    const Pipeline* pipeline = elempack == 4 ? pipeline_instancenorm_reduce_sum4_fp32_pack4[pb % 2] : pipeline_instancenorm_reduce_sum4_fp32[pb % 2];

                    cmd.record_pipeline(pipeline, bindings, constants, sum_workspace_reduced);

                    pb++;
                }

                sum_workspace = sum_workspace_reduced;
            }

            {
                std::vector<VkMat> bindings(2);
                bindings[0] = sum_workspace;
                bindings[1] = mean_workspace;

                std::vector<vk_constant_type> constants(5);
                constants[0].i = sum_workspace.w;
                constants[1].i = 1;
                constants[2].i = sum_workspace.c;
                constants[3].i = sum_workspace.cstep;
                constants[4].f = size;

                const Pipeline* pipeline = elempack == 4 ? pipeline_instancenorm_reduce_mean_pack4 : pipeline_instancenorm_reduce_mean;

                cmd.record_pipeline(pipeline, bindings, constants, mean_workspace);
            }
        }

        // var
        {
            // sub mean and square
            VkMat square_workspace;
            if (!use_radish_norm_sum4)
            {
                square_workspace.create(w, h, c, 4u * elempack, elempack, opt.workspace_vkallocator);
                std::vector<VkMat> bindings(3);
                bindings[0] = bottom_blob;
                bindings[1] = mean_workspace;
                bindings[2] = square_workspace;

                std::vector<vk_constant_type> constants(10);
                constants[0].i = std::min(3, bottom_blob.dims);
                constants[1].i = bottom_blob.w;
                constants[2].i = h;
                constants[3].i = bottom_blob.c;
                constants[4].i = bottom_blob.cstep;
                constants[5].i = square_workspace.dims;
                constants[6].i = square_workspace.w;
                constants[7].i = square_workspace.h;
                constants[8].i = square_workspace.c;
                constants[9].i = square_workspace.cstep;

                const Pipeline* pipeline = elempack == 4 ? pipeline_instancenorm_sub_mean_square_pack4 : pipeline_instancenorm_sub_mean_square;

                cmd.record_pipeline(pipeline, bindings, constants, square_workspace);
            }

            // reduce square
            VkMat sqsum_workspace;
            {
                int reduced_w = (size + 3) / 4;
                int reduced_h = 1;
                int reduced_c = use_radish_norm_sum4 ? bottom_blob.c : square_workspace.c;

                sqsum_workspace.create(reduced_w, reduced_h, reduced_c, 4u * elempack, elempack, opt.workspace_vkallocator);
                if (use_radish_norm_sum4 && sqsum_workspace.empty())
                    return -100;

                {
                    std::vector<VkMat> bindings(use_radish_norm_sum4 ? 3 : 2);
                    bindings[0] = use_radish_norm_sum4 ? bottom_blob : square_workspace;
                    bindings[1] = sqsum_workspace;
                    if (use_radish_norm_sum4)
                        bindings[2] = mean_workspace;

                    std::vector<vk_constant_type> constants(8);
                    constants[0].i = size;
                    constants[1].i = 1;
                    constants[2].i = use_radish_norm_sum4 ? bottom_blob.c : square_workspace.c;
                    constants[3].i = use_radish_norm_sum4 ? bottom_blob.cstep : square_workspace.cstep;
                    constants[4].i = sqsum_workspace.w;
                    constants[5].i = 1;
                    constants[6].i = sqsum_workspace.c;
                    constants[7].i = sqsum_workspace.cstep;

                    const Pipeline* pipeline = use_radish_norm_sum4 ? pipeline_instancenorm_radish_sum4_pack4 : (elempack == 4 ? pipeline_instancenorm_reduce_sum4_fp32_pack4[0] : pipeline_instancenorm_reduce_sum4_fp32[0]);

                    cmd.record_pipeline(pipeline, bindings, constants, sqsum_workspace);
                }
            }

            int pb = 1;
            while (sqsum_workspace.w > 4)
            {
                int reduced_w = (sqsum_workspace.w + 3) / 4;
                int reduced_h = 1;
                int reduced_c = sqsum_workspace.c;

                VkMat sqsum_workspace_reduced;
                sqsum_workspace_reduced.create(reduced_w, reduced_h, reduced_c, 4u * elempack, elempack, opt.workspace_vkallocator);
                if (coefficients_out && sqsum_workspace_reduced.empty())
                    return -100;

                {
                    std::vector<VkMat> bindings(2);
                    bindings[0] = sqsum_workspace;
                    bindings[1] = sqsum_workspace_reduced;

                    std::vector<vk_constant_type> constants(8);
                    constants[0].i = sqsum_workspace.w;
                    constants[1].i = 1;
                    constants[2].i = sqsum_workspace.c;
                    constants[3].i = sqsum_workspace.cstep;
                    constants[4].i = sqsum_workspace_reduced.w;
                    constants[5].i = 1;
                    constants[6].i = sqsum_workspace_reduced.c;
                    constants[7].i = sqsum_workspace_reduced.cstep;

                    const Pipeline* pipeline = elempack == 4 ? pipeline_instancenorm_reduce_sum4_fp32_pack4[pb % 2] : pipeline_instancenorm_reduce_sum4_fp32[pb % 2];

                    cmd.record_pipeline(pipeline, bindings, constants, sqsum_workspace_reduced);

                    pb++;
                }

                sqsum_workspace = sqsum_workspace_reduced;
            }

            {
                std::vector<VkMat> bindings(2);
                bindings[0] = sqsum_workspace;
                bindings[1] = var_workspace;

                std::vector<vk_constant_type> constants(5);
                constants[0].i = sqsum_workspace.w;
                constants[1].i = 1;
                constants[2].i = sqsum_workspace.c;
                constants[3].i = sqsum_workspace.cstep;
                constants[4].f = size;

                const Pipeline* pipeline = elempack == 4 ? pipeline_instancenorm_reduce_mean_pack4 : pipeline_instancenorm_reduce_mean;

                cmd.record_pipeline(pipeline, bindings, constants, var_workspace);
            }
        }
    }

    // coeffs
    VkMat coeffs_workspace;
    coeffs_workspace.create(c, elemsize * 2, elempack * 2, opt.workspace_vkallocator);
    if (coefficients_out && coeffs_workspace.empty())
        return -100;
    {
        std::vector<VkMat> bindings(5);
        bindings[0] = coeffs_workspace;
        bindings[1] = mean_workspace;
        bindings[2] = var_workspace;
        bindings[3] = gamma_data_gpu;
        bindings[4] = beta_data_gpu;

        std::vector<vk_constant_type> constants(1);
        constants[0].i = c;

        const Pipeline* pipeline = elempack == 4 ? pipeline_instancenorm_coeffs_pack4 : pipeline_instancenorm_coeffs;

        cmd.record_pipeline(pipeline, bindings, constants, coeffs_workspace);
    }

    if (coefficients_out)
    {
        *coefficients_out = coeffs_workspace;
        return 0;
    }

    // norm
    {
        std::vector<VkMat> bindings(outplace ? 3 : 2);
        bindings[0] = bottom_blob;
        bindings[1] = coeffs_workspace;
        if (outplace)
            bindings[2] = top_blob;

        std::vector<vk_constant_type> constants(5);
        constants[0].i = std::min(3, bottom_blob.dims);
        constants[1].i = bottom_blob.w;
        constants[2].i = h;
        constants[3].i = bottom_blob.c;
        constants[4].i = bottom_blob.cstep;

        const Pipeline* pipeline = outplace ? pipeline_instancenorm_radish_outplace_pack4 : (elempack == 4 ? pipeline_instancenorm_norm_pack4 : pipeline_instancenorm_norm);

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);
    }

    return 0;
}

} // namespace ncnn
