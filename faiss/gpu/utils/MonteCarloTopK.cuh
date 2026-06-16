/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceTensor.cuh>

namespace faiss {
namespace gpu {

/// Select top-k values from a full score matrix using a Monte-Carlo threshold.
///
/// `scores` is [queries, items]. `sampleScores` is [queries, samples] and is
/// used only to estimate the per-query threshold. For min selection (`dir ==
/// false`), the threshold is the `sampleK`-th smallest sample score and
/// candidates satisfy score <= threshold. For max selection (`dir == true`),
/// the threshold is the `sampleK`-th largest sample score and candidates satisfy
/// score >= threshold.
///
/// If a row has fewer than k candidates or more than `candidateCap` candidates,
/// the row falls back to exact top-k over `scores`.
void runMonteCarloTopKFromScores(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<float, 2, true>& scores,
        Tensor<float, 2, true>& sampleScores,
        int sampleK,
        int k,
        int candidateCap,
        bool dir,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        Tensor<int, 1, true>* outCounts = nullptr);

} // namespace gpu
} // namespace faiss
