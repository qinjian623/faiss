/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceTensor.cuh>

#include <cstdint>

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
/// By default, if a row has fewer than k candidates or more than
/// `candidateCap` candidates, the row falls back to exact top-k over `scores`.
/// If `overflowCutoff` is true, overflow rows keep a pseudo-random subset of
/// `candidateCap` candidates and do not fall back. Underflow rows still fall
/// back because they cannot produce k valid outputs.
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
        Tensor<int, 1, true>* outCounts = nullptr,
        bool overflowCutoff = false,
        uint64_t cutoffSeed = 0);

} // namespace gpu
} // namespace faiss
