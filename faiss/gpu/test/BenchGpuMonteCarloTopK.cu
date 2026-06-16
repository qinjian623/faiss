/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/BlockSelectKernel.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/MonteCarloTopK.cuh>
#include <faiss/gpu/utils/StaticUtils.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct BenchCase {
    int rows;
    int cols;
    int k;
    int sampleSize;
    int sampleK;
    int candidateTarget;
    int candidateCap;
    int warmup;
    int iters;
};

__global__ void fillScoresKernel(faiss::gpu::Tensor<float, 2, true> scores) {
    auto row = faiss::idx_t(blockIdx.y);
    auto col = faiss::idx_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (row >= scores.getSize(0) || col >= scores.getSize(1)) {
        return;
    }

    int cols = static_cast<int>(scores.getSize(1));
    int permuted = static_cast<int>((col * 37 + row * 17) % cols);
    scores[row][col] = static_cast<float>(permuted) +
            static_cast<float>(row) * 0.000001f;
}

__global__ void fillSampleScoresKernel(
        faiss::gpu::Tensor<float, 2, true> sampleScores,
        int sampleK,
        int candidateTarget,
        int cols) {
    auto row = faiss::idx_t(blockIdx.y);
    auto col = faiss::idx_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (row >= sampleScores.getSize(0) ||
        col >= sampleScores.getSize(1)) {
        return;
    }

    float rowOffset = static_cast<float>(row) * 0.000001f;
    if (col < sampleK - 1) {
        sampleScores[row][col] = -1000000.0f + static_cast<float>(col);
    } else if (col == sampleK - 1) {
        sampleScores[row][col] =
                static_cast<float>(candidateTarget - 1) + rowOffset;
    } else {
        sampleScores[row][col] =
                static_cast<float>(cols + col) + rowOffset;
    }
}

template <typename Fn>
float timeLoop(cudaStream_t stream, int warmup, int iters, Fn fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    CUDA_VERIFY(cudaStreamSynchronize(stream));

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_VERIFY(cudaEventCreate(&start));
    CUDA_VERIFY(cudaEventCreate(&stop));
    CUDA_VERIFY(cudaEventRecord(start, stream));

    for (int i = 0; i < iters; ++i) {
        fn();
    }

    CUDA_VERIFY(cudaEventRecord(stop, stream));
    CUDA_VERIFY(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_VERIFY(cudaEventElapsedTime(&ms, start, stop));
    CUDA_VERIFY(cudaEventDestroy(start));
    CUDA_VERIFY(cudaEventDestroy(stop));

    return ms / static_cast<float>(iters);
}

void verifySame(
        faiss::gpu::DeviceTensor<float, 2, true>& exactDistances,
        faiss::gpu::DeviceTensor<faiss::idx_t, 2, true>& exactIndices,
        faiss::gpu::DeviceTensor<float, 2, true>& mcDistances,
        faiss::gpu::DeviceTensor<faiss::idx_t, 2, true>& mcIndices,
        cudaStream_t stream) {
    faiss::gpu::HostTensor<float, 2, true> exactHostDistances(
            exactDistances, stream);
    faiss::gpu::HostTensor<faiss::idx_t, 2, true> exactHostIndices(
            exactIndices, stream);
    faiss::gpu::HostTensor<float, 2, true> mcHostDistances(mcDistances, stream);
    faiss::gpu::HostTensor<faiss::idx_t, 2, true> mcHostIndices(
            mcIndices, stream);

    for (int row = 0; row < exactHostDistances.getSize(0); ++row) {
        for (int rank = 0; rank < exactHostDistances.getSize(1); ++rank) {
            if (exactHostDistances[row][rank] != mcHostDistances[row][rank] ||
                exactHostIndices[row][rank] != mcHostIndices[row][rank]) {
                std::cerr << "mismatch row=" << row << " rank=" << rank
                          << " exact=(" << exactHostDistances[row][rank]
                          << ", " << exactHostIndices[row][rank] << ")"
                          << " mc=(" << mcHostDistances[row][rank] << ", "
                          << mcHostIndices[row][rank] << ")" << std::endl;
                std::exit(2);
            }
        }
    }
}

void runBenchCase(const BenchCase& c) {
    using namespace faiss;
    using namespace faiss::gpu;

    StandardGpuResources res;
    auto resources = res.getResources().get();
    auto stream = resources->getDefaultStreamCurrentDevice();

    DeviceTensor<float, 2, true> scores(
            resources, makeDevAlloc(AllocType::Other, 0), {c.rows, c.cols});
    DeviceTensor<float, 2, true> sampleScores(
            resources,
            makeDevAlloc(AllocType::Other, 0),
            {c.rows, c.sampleSize});
    DeviceTensor<float, 2, true> exactDistances(
            resources, makeDevAlloc(AllocType::Other, 0), {c.rows, c.k});
    DeviceTensor<idx_t, 2, true> exactIndices(
            resources, makeDevAlloc(AllocType::Other, 0), {c.rows, c.k});
    DeviceTensor<float, 2, true> mcDistances(
            resources, makeDevAlloc(AllocType::Other, 0), {c.rows, c.k});
    DeviceTensor<idx_t, 2, true> mcIndices(
            resources, makeDevAlloc(AllocType::Other, 0), {c.rows, c.k});
    DeviceTensor<int, 1, true> counts(
            resources, makeDevAlloc(AllocType::Other, 0), {c.rows});

    constexpr int threads = 256;
    dim3 scoreBlocks(
            utils::divUp(idx_t(c.cols), idx_t(threads)),
            static_cast<unsigned int>(c.rows));
    fillScoresKernel<<<scoreBlocks, threads, 0, stream>>>(scores);

    dim3 sampleBlocks(
            utils::divUp(idx_t(c.sampleSize), idx_t(threads)),
            static_cast<unsigned int>(c.rows));
    fillSampleScoresKernel<<<sampleBlocks, threads, 0, stream>>>(
            sampleScores, c.sampleK, c.candidateTarget, c.cols);
    CUDA_VERIFY(cudaStreamSynchronize(stream));

    runBlockSelect(scores, exactDistances, exactIndices, false, c.k, stream);
    runMonteCarloTopKFromScores(
            resources,
            stream,
            scores,
            sampleScores,
            c.sampleK,
            c.k,
            c.candidateCap,
            false,
            mcDistances,
            mcIndices,
            &counts);
    CUDA_VERIFY(cudaStreamSynchronize(stream));
    verifySame(exactDistances, exactIndices, mcDistances, mcIndices, stream);

    HostTensor<int, 1, true> hostCounts(counts, stream);
    int minCount = hostCounts.data()[0];
    int maxCount = hostCounts.data()[0];
    long long sumCount = 0;
    for (int row = 0; row < c.rows; ++row) {
        int count = hostCounts.data()[row];
        minCount = std::min(minCount, count);
        maxCount = std::max(maxCount, count);
        sumCount += count;
    }

    auto exactMs = timeLoop(stream, c.warmup, c.iters, [&] {
        runBlockSelect(scores, exactDistances, exactIndices, false, c.k, stream);
    });

    auto mcMs = timeLoop(stream, c.warmup, c.iters, [&] {
        runMonteCarloTopKFromScores(
                resources,
                stream,
                scores,
                sampleScores,
                c.sampleK,
                c.k,
                c.candidateCap,
                false,
                mcDistances,
                mcIndices,
                nullptr);
    });

    double speedup = static_cast<double>(exactMs) / static_cast<double>(mcMs);
    double avgCount = static_cast<double>(sumCount) /
            static_cast<double>(std::max(1, c.rows));

    std::cout << std::setw(6) << c.rows << " " << std::setw(8) << c.cols
              << " " << std::setw(3) << c.k << " " << std::setw(5)
              << c.sampleSize << " " << std::setw(5) << c.sampleK << " "
              << std::setw(6) << c.candidateTarget << " " << std::setw(6)
              << c.candidateCap << " " << std::fixed << std::setprecision(3)
              << std::setw(10) << exactMs << " " << std::setw(10) << mcMs
              << " " << std::setw(7) << speedup << " " << std::setw(8)
              << avgCount << " " << minCount << ".." << maxCount
              << std::endl;
}

} // namespace

int main() {
    std::vector<BenchCase> cases = {
            {128, 16384, 10, 128, 16, 64, 128, 5, 30},
            {512, 65536, 10, 128, 16, 64, 128, 5, 20},
            {512, 262144, 10, 128, 16, 64, 128, 3, 10},
            {1024, 262144, 10, 128, 16, 64, 128, 3, 10},
    };

    int device = 0;
    CUDA_VERIFY(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_VERIFY(cudaGetDeviceProperties(&prop, device));

    std::cout << "device=" << prop.name << " cases=" << cases.size()
              << std::endl;
    std::cout << "  rows     cols   k sample    sk target    cap   exact_ms"
                 "      mc_ms speedup avg_cnt min..max"
              << std::endl;

    for (const auto& c : cases) {
        runBenchCase(c);
    }

    return 0;
}
