// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONVOLUTION1D_VULKAN_H
#define LAYER_CONVOLUTION1D_VULKAN_H

#include "convolution1d.h"

namespace ncnn {

class Convolution1D_vulkan : public Convolution1D
{
public:
    Convolution1D_vulkan();

    virtual int load_param(const ParamDict& pd);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using Convolution1D::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    ncnn::Layer* padding;

    Mat weight_data_packed;

    VkMat weight_data_gpu;
    VkMat bias_data_gpu;

    Pipeline* pipeline_convolution1d;
    Pipeline* pipeline_convolution1d_1x1s1d1;
    Pipeline* pipeline_convolution1d_gemm;
    Pipeline* pipeline_radish_planar_pad;

    bool use_radish_tiled;
    bool use_radish_gemm;
    int radish_gemm_rows;
    bool use_radish_wino;
    Pipeline* pipeline_radish_wino_in;
    Pipeline* pipeline_radish_wino_out;
    bool radish_tiled32;
    bool radish_planar;
    bool radish_pad_channels;
    bool radish_direct_window;
    bool radish_direct_weight_half;

    // cooperative matrix
    bool use_cooperative_matrix;
    int coopmat_M;
    int coopmat_N;
    int coopmat_K;
    int coopmat_subgroup_size;
    int UNROLL_SG_M;
    int UNROLL_SG_N;
    int UNROLL_SG_K;
    int UNROLL_WG_M;
    int UNROLL_WG_N;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTION1D_VULKAN_H
