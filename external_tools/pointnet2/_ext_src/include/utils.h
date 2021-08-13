// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

// !! we update this source file to support PyTorch 1.5+ according to
// https://github.com/erikwijmans/Pointnet2_PyTorch/commit/1d5dca2673ee2831b9b01efe597b2ba8d12726f1 !!

#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define CHECK_CUDA(x)                                          \
  do {                                                         \
    AT_ASSERT(x.is_cuda(), #x " must be a CUDA tensor"); \
  } while (0)

#define CHECK_CONTIGUOUS(x)                                         \
  do {                                                              \
    AT_ASSERT(x.is_contiguous(), #x " must be a contiguous tensor"); \
  } while (0)

#define CHECK_IS_INT(x)                              \
  do {                                               \
    AT_ASSERT(x.scalar_type() == at::ScalarType::Int, \
             #x " must be an int tensor");           \
  } while (0)

#define CHECK_IS_FLOAT(x)                              \
  do {                                                 \
    AT_ASSERT(x.scalar_type() == at::ScalarType::Float, \
             #x " must be a float tensor");            \
  } while (0)
