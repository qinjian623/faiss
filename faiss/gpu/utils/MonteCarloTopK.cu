/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/impl/IndexUtils.h>
#include <faiss/gpu/utils/BlockSelectKernel.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/MonteCarloTopK.cuh>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {
namespace gpu {

namespace {

constexpr int kFilterThreads = 256;
constexpr int kFallbackThreads = 256;

__device__ __forceinline__ float mcSentinel(bool dir) {
    return dir ? Limits<float>::getMin() : Limits<float>::getMax();
}

__device__ __forceinline__ bool mcBetter(
        float value,
        idx_t index,
        float currentValue,
        idx_t currentIndex,
        bool dir) {
    if (currentIndex < 0) {
        return true;
    }

    if (dir) {
        return value > currentValue ||
                (value == currentValue && index < currentIndex);
    } else {
        return value < currentValue ||
                (value == currentValue && index < currentIndex);
    }
}

__global__ void mcFilterKernel(
        Tensor<float, 2, true> scores,
        Tensor<float, 2, true> thresholdValues,
        int sampleK,
        Tensor<float, 2, true> candidateValues,
        Tensor<idx_t, 2, true> candidateIndices,
        Tensor<int, 1, true> counts,
        bool dir) {
    auto row = idx_t(blockIdx.y);
    auto col = idx_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (row >= scores.getSize(0) || col >= scores.getSize(1)) {
        return;
    }

    float threshold = thresholdValues[row][sampleK - 1];
    float value = scores[row][col];
    bool keep = dir ? value >= threshold : value <= threshold;

    if (keep) {
        int pos = atomicAdd(counts.data() + row, 1);
        if (pos < candidateValues.getSize(1)) {
            candidateValues[row][pos] = value;
            candidateIndices[row][pos] = col;
        }
    }
}

__global__ void mcPadCandidatesKernel(
        Tensor<float, 2, true> candidateValues,
        Tensor<idx_t, 2, true> candidateIndices,
        Tensor<int, 1, true> counts,
        bool dir) {
    auto row = idx_t(blockIdx.y);
    auto col = idx_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (row >= candidateValues.getSize(0) ||
        col >= candidateValues.getSize(1)) {
        return;
    }

    int validCount = counts[row];
    if (validCount > candidateValues.getSize(1)) {
        validCount = candidateValues.getSize(1);
    }

    if (col >= validCount) {
        candidateValues[row][col] = mcSentinel(dir);
        candidateIndices[row][col] = -1;
    }
}

__global__ void mcMapCandidateIndicesKernel(
        Tensor<idx_t, 2, true> candidatePositions,
        Tensor<idx_t, 2, true> candidateIndices,
        Tensor<idx_t, 2, true> outIndices) {
    auto row = idx_t(blockIdx.y);
    auto col = idx_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (row >= outIndices.getSize(0) || col >= outIndices.getSize(1)) {
        return;
    }

    idx_t candidatePos = candidatePositions[row][col];
    outIndices[row][col] = candidatePos >= 0 &&
                    candidatePos < candidateIndices.getSize(1)
            ? candidateIndices[row][candidatePos]
            : -1;
}

__global__ void mcFallbackExactTopKKernel(
        Tensor<float, 2, true> scores,
        Tensor<int, 1, true> counts,
        int candidateCap,
        int k,
        bool dir,
        Tensor<float, 2, true> outDistances,
        Tensor<idx_t, 2, true> outIndices) {
    extern __shared__ unsigned char shared[];
    float* sharedValues = reinterpret_cast<float*>(shared);
    idx_t* sharedIndices =
            reinterpret_cast<idx_t*>(sharedValues + blockDim.x);

    auto row = idx_t(blockIdx.x);
    int tid = threadIdx.x;
    int count = counts[row];

    if (count >= k && count <= candidateCap) {
        return;
    }

    for (int rank = 0; rank < k; ++rank) {
        float bestValue = mcSentinel(dir);
        idx_t bestIndex = -1;

        for (idx_t col = tid; col < scores.getSize(1); col += blockDim.x) {
            bool used = false;
            for (int prev = 0; prev < rank; ++prev) {
                if (outIndices[row][prev] == col) {
                    used = true;
                    break;
                }
            }
            if (used) {
                continue;
            }

            float value = scores[row][col];
            if (mcBetter(value, col, bestValue, bestIndex, dir)) {
                bestValue = value;
                bestIndex = col;
            }
        }

        sharedValues[tid] = bestValue;
        sharedIndices[tid] = bestIndex;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride &&
                mcBetter(sharedValues[tid + stride],
                         sharedIndices[tid + stride],
                         sharedValues[tid],
                         sharedIndices[tid],
                         dir)) {
                sharedValues[tid] = sharedValues[tid + stride];
                sharedIndices[tid] = sharedIndices[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            outDistances[row][rank] = sharedValues[0];
            outIndices[row][rank] = sharedIndices[0];
        }
        __syncthreads();
    }
}

} // namespace

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
        Tensor<int, 1, true>* outCounts) {
    auto rows = scores.getSize(0);
    auto cols = scores.getSize(1);

    FAISS_THROW_IF_NOT(rows > 0 && cols > 0);
    FAISS_THROW_IF_NOT(scores.getSize(0) == sampleScores.getSize(0));
    FAISS_THROW_IF_NOT(outDistances.getSize(0) == rows);
    FAISS_THROW_IF_NOT(outIndices.getSize(0) == rows);
    FAISS_THROW_IF_NOT(outDistances.getSize(1) == k);
    FAISS_THROW_IF_NOT(outIndices.getSize(1) == k);
    FAISS_THROW_IF_NOT(k > 0 && k <= cols);
    FAISS_THROW_IF_NOT(k <= getMaxKSelection(false));
    FAISS_THROW_IF_NOT(sampleK > 0 && sampleK <= sampleScores.getSize(1));
    FAISS_THROW_IF_NOT(sampleK <= getMaxKSelection(false));
    FAISS_THROW_IF_NOT(candidateCap >= k);
    FAISS_THROW_IF_NOT(cols <= static_cast<idx_t>(Limits<int>::getMax()));
    FAISS_THROW_IF_NOT(outCounts == nullptr || outCounts->getSize(0) == rows);

    DeviceTensor<float, 2, true> thresholdValues(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {rows, sampleK});
    DeviceTensor<idx_t, 2, true> thresholdIndices(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {rows, sampleK});
    runBlockSelect(
            sampleScores,
            thresholdValues,
            thresholdIndices,
            dir,
            sampleK,
            stream);

    DeviceTensor<float, 2, true> candidateValues(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {rows, candidateCap});
    DeviceTensor<idx_t, 2, true> candidateIndices(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {rows, candidateCap});
    DeviceTensor<int, 1, true> counts(
            res, makeTempAlloc(AllocType::Other, stream), {rows});
    counts.zero(stream);

    dim3 filterBlocks(
            utils::divUp(cols, idx_t(kFilterThreads)),
            static_cast<unsigned int>(rows));
    mcFilterKernel<<<filterBlocks, kFilterThreads, 0, stream>>>(
            scores,
            thresholdValues,
            sampleK,
            candidateValues,
            candidateIndices,
            counts,
            dir);

    dim3 candidateBlocks(
            utils::divUp(idx_t(candidateCap), idx_t(kFilterThreads)),
            static_cast<unsigned int>(rows));
    mcPadCandidatesKernel<<<candidateBlocks, kFilterThreads, 0, stream>>>(
            candidateValues, candidateIndices, counts, dir);

    DeviceTensor<idx_t, 2, true> candidatePositions(
            res, makeTempAlloc(AllocType::Other, stream), {rows, k});
    runBlockSelect(
            candidateValues,
            outDistances,
            candidatePositions,
            dir,
            k,
            stream);

    dim3 mapBlocks(
            utils::divUp(idx_t(k), idx_t(kFilterThreads)),
            static_cast<unsigned int>(rows));
    mcMapCandidateIndicesKernel<<<mapBlocks, kFilterThreads, 0, stream>>>(
            candidatePositions, candidateIndices, outIndices);

    size_t sharedBytes = kFallbackThreads * (sizeof(float) + sizeof(idx_t));
    mcFallbackExactTopKKernel<<<
            static_cast<unsigned int>(rows),
            kFallbackThreads,
            sharedBytes,
            stream>>>(
            scores, counts, candidateCap, k, dir, outDistances, outIndices);

    if (outCounts) {
        outCounts->copyFrom(counts, stream);
    }
}

} // namespace gpu
} // namespace faiss
