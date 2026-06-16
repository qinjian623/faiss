/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/MonteCarloTopK.cuh>
#include <gtest/gtest.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace {

void fillScores(faiss::gpu::HostTensor<float, 2, true>& scores) {
    for (int row = 0; row < scores.getSize(0); ++row) {
        for (int col = 0; col < scores.getSize(1); ++col) {
            int permuted = (col * 37 + row * 11) % scores.getSize(1);
            scores[row][col] = static_cast<float>(permuted) +
                    static_cast<float>(row) * 0.001f;
        }
    }
}

std::vector<std::pair<float, faiss::idx_t>> exactTopK(
        faiss::gpu::HostTensor<float, 2, true>& scores,
        int row,
        int k,
        bool dir) {
    std::vector<std::pair<float, faiss::idx_t>> values;
    for (int col = 0; col < scores.getSize(1); ++col) {
        values.emplace_back(scores[row][col], col);
    }

    std::sort(
            values.begin(),
            values.end(),
            [dir](const auto& a, const auto& b) {
                if (a.first == b.first) {
                    return a.second < b.second;
                }
                return dir ? a.first > b.first : a.first < b.first;
            });

    values.resize(k);
    return values;
}

void assertMatchesExact(
        faiss::gpu::HostTensor<float, 2, true>& scores,
        faiss::gpu::HostTensor<float, 2, true>& outDistances,
        faiss::gpu::HostTensor<faiss::idx_t, 2, true>& outIndices,
        int k,
        bool dir) {
    for (int row = 0; row < scores.getSize(0); ++row) {
        auto exact = exactTopK(scores, row, k, dir);
        for (int i = 0; i < k; ++i) {
            EXPECT_EQ(outDistances[row][i], exact[i].first)
                    << "row " << row << " rank " << i;
            EXPECT_EQ(outIndices[row][i], exact[i].second)
                    << "row " << row << " rank " << i;
        }
    }
}

void runCase(
        bool dir,
        const std::vector<float>& sampleTemplate,
        int sampleK,
        int candidateCap) {
    using namespace faiss;
    using namespace faiss::gpu;

    constexpr int rows = 5;
    constexpr int cols = 257;
    constexpr int k = 5;

    StandardGpuResources res;
    auto resources = res.getResources().get();
    auto stream = resources->getDefaultStreamCurrentDevice();

    HostTensor<float, 2, true> hostScores({rows, cols});
    fillScores(hostScores);

    HostTensor<float, 2, true> hostSampleScores(
            {rows, static_cast<int>(sampleTemplate.size())});
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < sampleTemplate.size(); ++col) {
            hostSampleScores[row][col] = sampleTemplate[col] +
                    static_cast<float>(row) * 0.001f;
        }
    }

    DeviceTensor<float, 2, true> gpuScores(
            resources, makeDevAlloc(AllocType::Other, 0), hostScores);
    DeviceTensor<float, 2, true> gpuSampleScores(
            resources, makeDevAlloc(AllocType::Other, 0), hostSampleScores);
    DeviceTensor<float, 2, true> gpuOutDistances(
            resources, makeDevAlloc(AllocType::Other, 0), {rows, k});
    DeviceTensor<idx_t, 2, true> gpuOutIndices(
            resources, makeDevAlloc(AllocType::Other, 0), {rows, k});
    DeviceTensor<int, 1, true> gpuCounts(
            resources, makeDevAlloc(AllocType::Other, 0), {rows});

    runMonteCarloTopKFromScores(
            resources,
            stream,
            gpuScores,
            gpuSampleScores,
            sampleK,
            k,
            candidateCap,
            dir,
            gpuOutDistances,
            gpuOutIndices,
            &gpuCounts);

    HostTensor<float, 2, true> outDistances(gpuOutDistances, stream);
    HostTensor<idx_t, 2, true> outIndices(gpuOutIndices, stream);
    assertMatchesExact(hostScores, outDistances, outIndices, k, dir);
}

} // namespace

TEST(TestGpuMonteCarloTopK, minCandidatePath) {
    runCase(false, {0.0f, 1.0f, 32.0f, 200.0f}, 3, 64);
}

TEST(TestGpuMonteCarloTopK, minFallbackWhenTooFewCandidates) {
    runCase(false, {-10.0f, -9.0f, -8.0f}, 1, 64);
}

TEST(TestGpuMonteCarloTopK, minFallbackWhenCandidateCapOverflows) {
    runCase(false, {0.0f, 128.0f, 255.0f}, 3, 8);
}

TEST(TestGpuMonteCarloTopK, minCutoffWhenCandidateCapOverflows) {
    using namespace faiss;
    using namespace faiss::gpu;

    constexpr int rows = 1;
    constexpr int cols = 4096;
    constexpr int k = 5;

    StandardGpuResources res;
    auto resources = res.getResources().get();
    auto stream = resources->getDefaultStreamCurrentDevice();

    HostTensor<float, 2, true> hostScores({rows, cols});
    for (int col = 0; col < cols; ++col) {
        hostScores[0][col] = static_cast<float>(col);
    }

    HostTensor<float, 2, true> hostSampleScores({rows, 1});
    hostSampleScores[0][0] = static_cast<float>(cols);

    DeviceTensor<float, 2, true> gpuScores(
            resources, makeDevAlloc(AllocType::Other, 0), hostScores);
    DeviceTensor<float, 2, true> gpuSampleScores(
            resources, makeDevAlloc(AllocType::Other, 0), hostSampleScores);
    DeviceTensor<float, 2, true> gpuOutDistances(
            resources, makeDevAlloc(AllocType::Other, 0), {rows, k});
    DeviceTensor<idx_t, 2, true> gpuOutIndices(
            resources, makeDevAlloc(AllocType::Other, 0), {rows, k});
    DeviceTensor<int, 1, true> gpuCounts(
            resources, makeDevAlloc(AllocType::Other, 0), {rows});

    runMonteCarloTopKFromScores(
            resources,
            stream,
            gpuScores,
            gpuSampleScores,
            1,
            k,
            k,
            false,
            gpuOutDistances,
            gpuOutIndices,
            &gpuCounts,
            true,
            123);

    HostTensor<idx_t, 2, true> outIndices(gpuOutIndices, stream);
    HostTensor<int, 1, true> counts(gpuCounts, stream);

    EXPECT_EQ(counts[0], cols);

    int exactHits = 0;
    for (int i = 0; i < k; ++i) {
        EXPECT_GE(outIndices[0][i], 0);
        EXPECT_LT(outIndices[0][i], cols);
        if (outIndices[0][i] < k) {
            ++exactHits;
        }
    }
    EXPECT_LT(exactHits, k);
}

TEST(TestGpuMonteCarloTopK, maxCandidatePath) {
    runCase(true, {256.0f, 255.0f, 220.0f, 0.0f}, 3, 64);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
