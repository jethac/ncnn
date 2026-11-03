// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INSTANCENORM_VULKAN_H
#define LAYER_INSTANCENORM_VULKAN_H

#include "instancenorm.h"

namespace ncnn {

class InstanceNorm_vulkan : public InstanceNorm
{
public:
    InstanceNorm_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    int forward_radish_coefficients(const VkMat& input, VkMat& coefficients, VkCompute& cmd, const Option& opt) const;

    using InstanceNorm::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

    using InstanceNorm::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

private:
    int forward_impl(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt, bool outplace, VkMat* coefficients_out = 0) const;

public:
    bool radish_norm_outplace;
    Pipeline* pipeline_instancenorm_radish_outplace_pack4;
    VkMat gamma_data_gpu;
    VkMat beta_data_gpu;

    Pipeline* pipeline_instancenorm_reduce_sum4_fp16_to_fp32;
    Pipeline* pipeline_instancenorm_reduce_sum4_fp32[2];
    Pipeline* pipeline_instancenorm_reduce_mean;
    Pipeline* pipeline_instancenorm_sub_mean_square;
    Pipeline* pipeline_instancenorm_coeffs;
    Pipeline* pipeline_instancenorm_norm;

    Pipeline* pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack4;
    Pipeline* pipeline_instancenorm_reduce_sum4_fp32_pack4[2];
    Pipeline* pipeline_instancenorm_reduce_mean_pack4;
    Pipeline* pipeline_instancenorm_sub_mean_square_pack4;
    Pipeline* pipeline_instancenorm_radish_sum4_pack4;
    Pipeline* pipeline_instancenorm_coeffs_pack4;
    Pipeline* pipeline_instancenorm_norm_pack4;

    Pipeline* pipeline_instancenorm_reduce_subgroup;
    Pipeline* pipeline_instancenorm_reduce_subgroup_pack4;
};

} // namespace ncnn

#endif // LAYER_INSTANCENORM_VULKAN_H
