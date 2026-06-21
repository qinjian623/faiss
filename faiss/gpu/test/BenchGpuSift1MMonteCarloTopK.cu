/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cublas_v2.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/BlockSelectKernel.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/MonteCarloTopK.cuh>
#include <faiss/gpu/utils/StaticUtils.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string dataDir = "sift1M";
    int queries = 1024;
    int batchSize = 32;
    int k = 64;
    int sampleN = 8192;
    int sampleK = 4;
    int candidateCap = 50000;
    int baseRepeat = 1;
    int warmup = 1;
    int repeats = 3;
    int seed = 0;
    bool overflowCutoff = false;
    uint64_t cutoffSeed = 0;
};

struct Matrix {
    int rows = 0;
    int dim = 0;
    std::vector<float> data;
};

struct IntMatrix {
    int rows = 0;
    int dim = 0;
    std::vector<int> data;
};

struct RunStats {
    double distanceS = 0.0;
    double sampleDistanceS = 0.0;
    double scoreRepeatS = 0.0;
    double exactSelectionS = 0.0;
    double mcSelectionS = 0.0;
    double recallVsExactSum = 0.0;
    double recallVsExactMin = 1.0;
    double scoreValidSum = 0.0;
    double scoreValidMin = 1.0;
    double tieAwareSum = 0.0;
    double tieAwareMin = 1.0;
    double exactGtSum = 0.0;
    double mcGtSum = 0.0;
    int qualityRows = 0;
    long long candidateSum = 0;
    int candidateMin = 0;
    int candidateMax = 0;
    long long boundaryTieSum = 0;
    int boundaryTieMax = 0;
    int boundaryTieRows = 0;
    int fallbacks = 0;
    int underflows = 0;
    int overflows = 0;
    std::vector<int> candidateCounts;
    std::vector<double> exactTotalBatchS;
    std::vector<double> mcTotalBatchS;
    std::vector<double> mcSelectionBatchS;
};

[[noreturn]] void fail(const std::string& msg) {
    std::cerr << msg << std::endl;
    std::exit(1);
}

void cublasCheck(cublasStatus_t status, const char* call) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream out;
        out << call << " failed with cublasStatus=" << int(status);
        fail(out.str());
    }
}

std::string joinPath(const std::string& dir, const std::string& name) {
    if (dir.empty() || dir.back() == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}

Matrix readFvecs(const std::string& path, int maxRows = 0) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail("failed to open " + path);
    }

    Matrix out;
    while (!in.eof()) {
        int32_t dim = 0;
        in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        if (!in) {
            break;
        }
        if (dim <= 0) {
            fail("invalid fvecs dimension in " + path);
        }
        if (out.dim == 0) {
            out.dim = dim;
        } else if (out.dim != dim) {
            fail("inconsistent fvecs dimension in " + path);
        }
        if (maxRows > 0 && out.rows >= maxRows) {
            in.seekg(sizeof(float) * dim, std::ios::cur);
            continue;
        }

        auto oldSize = out.data.size();
        out.data.resize(oldSize + dim);
        in.read(reinterpret_cast<char*>(out.data.data() + oldSize),
                sizeof(float) * dim);
        if (!in) {
            fail("truncated fvecs record in " + path);
        }
        ++out.rows;
    }

    return out;
}

IntMatrix readIvecs(const std::string& path, int maxRows = 0) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail("failed to open " + path);
    }

    IntMatrix out;
    while (!in.eof()) {
        int32_t dim = 0;
        in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        if (!in) {
            break;
        }
        if (dim <= 0) {
            fail("invalid ivecs dimension in " + path);
        }
        if (out.dim == 0) {
            out.dim = dim;
        } else if (out.dim != dim) {
            fail("inconsistent ivecs dimension in " + path);
        }
        if (maxRows > 0 && out.rows >= maxRows) {
            in.seekg(sizeof(int32_t) * dim, std::ios::cur);
            continue;
        }

        auto oldSize = out.data.size();
        out.data.resize(oldSize + dim);
        in.read(reinterpret_cast<char*>(out.data.data() + oldSize),
                sizeof(int32_t) * dim);
        if (!in) {
            fail("truncated ivecs record in " + path);
        }
        ++out.rows;
    }

    return out;
}

std::vector<float> rowNorms(const Matrix& matrix) {
    std::vector<float> norms(matrix.rows);
    for (int row = 0; row < matrix.rows; ++row) {
        double sum = 0.0;
        for (int d = 0; d < matrix.dim; ++d) {
            float v = matrix.data[size_t(row) * matrix.dim + d];
            sum += double(v) * double(v);
        }
        norms[row] = static_cast<float>(sum);
    }
    return norms;
}

Matrix gatherRows(const Matrix& matrix, const std::vector<int>& ids) {
    Matrix out;
    out.rows = static_cast<int>(ids.size());
    out.dim = matrix.dim;
    out.data.resize(size_t(out.rows) * out.dim);
    for (int row = 0; row < out.rows; ++row) {
        std::copy_n(matrix.data.data() + size_t(ids[row]) * matrix.dim,
                    matrix.dim,
                    out.data.data() + size_t(row) * out.dim);
    }
    return out;
}

std::vector<int> sampleIds(int n, int sampleN, int seed) {
    std::vector<int> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(ids.begin(), ids.end(), rng);
    ids.resize(sampleN);
    return ids;
}

template <typename T, int Dim>
faiss::gpu::HostTensor<T, Dim, true> hostTensorFromVector(
        const std::vector<T>& data,
        std::initializer_list<faiss::idx_t> sizes) {
    faiss::gpu::HostTensor<T, Dim, true> host(sizes);
    std::copy(data.begin(), data.end(), host.data());
    return host;
}

__global__ void l2FromDotKernel(
        faiss::gpu::Tensor<float, 2, true> scores,
        faiss::gpu::Tensor<float, 1, true> norms) {
    auto row = faiss::idx_t(blockIdx.y);
    auto col = faiss::idx_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (row >= scores.getSize(0) || col >= scores.getSize(1)) {
        return;
    }

    scores[row][col] = norms[col] - 2.0f * scores[row][col];
}

__global__ void repeatScoresKernel(
        faiss::gpu::Tensor<float, 2, true> in,
        faiss::gpu::Tensor<float, 2, true> out) {
    auto row = faiss::idx_t(blockIdx.y);
    auto col = faiss::idx_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (row >= out.getSize(0) || col >= out.getSize(1)) {
        return;
    }

    out[row][col] = in[row][col % in.getSize(1)].data()[0];
}

__global__ void boundaryTieCountKernel(
        faiss::gpu::Tensor<float, 2, true> scores,
        faiss::gpu::Tensor<float, 2, true> exactDistances,
        faiss::gpu::Tensor<int, 1, true> tieCounts,
        int k,
        float eps) {
    auto row = faiss::idx_t(blockIdx.x);
    if (row >= scores.getSize(0)) {
        return;
    }

    __shared__ float boundary;
    if (threadIdx.x == 0) {
        float b = exactDistances[row][0].data()[0];
        for (int i = 1; i < k; ++i) {
            b = fmaxf(b, exactDistances[row][i].data()[0]);
        }
        boundary = b;
        tieCounts[row] = 0;
    }
    __syncthreads();

    int local = 0;
    for (auto col = faiss::idx_t(threadIdx.x); col < scores.getSize(1);
         col += blockDim.x) {
        float v = scores[row][col].data()[0];
        if (fabsf(v - boundary) <= eps) {
            ++local;
        }
    }
    if (local > 0) {
        atomicAdd(tieCounts[row].data(), local);
    }
}

template <typename Fn>
float timeOnce(cudaStream_t stream, Fn fn) {
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_VERIFY(cudaEventCreate(&start));
    CUDA_VERIFY(cudaEventCreate(&stop));
    CUDA_VERIFY(cudaEventRecord(start, stream));
    fn();
    CUDA_VERIFY(cudaEventRecord(stop, stream));
    CUDA_VERIFY(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_VERIFY(cudaEventElapsedTime(&ms, start, stop));
    CUDA_VERIFY(cudaEventDestroy(start));
    CUDA_VERIFY(cudaEventDestroy(stop));
    return ms / 1000.0f;
}

void computeScores(
        cublasHandle_t handle,
        cudaStream_t stream,
        faiss::gpu::Tensor<float, 2, true>& queries,
        faiss::gpu::Tensor<float, 2, true>& base,
        faiss::gpu::Tensor<float, 1, true>& baseNorms,
        faiss::gpu::Tensor<float, 2, true>& scores) {
    int rows = static_cast<int>(queries.getSize(0));
    int cols = static_cast<int>(base.getSize(0));
    int dim = static_cast<int>(queries.getSize(1));
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasCheck(
            cublasSgemm(
                    handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    cols,
                    rows,
                    dim,
                    &alpha,
                    base.data(),
                    dim,
                    queries.data(),
                    dim,
                    &beta,
                    scores.data(),
                    cols),
            "cublasSgemm");

    constexpr int threads = 256;
    dim3 blocks(
            faiss::gpu::utils::divUp(
                    faiss::idx_t(cols), faiss::idx_t(threads)),
            static_cast<unsigned int>(rows));
    l2FromDotKernel<<<blocks, threads, 0, stream>>>(scores, baseNorms);
}

double overlapAtK(
        const faiss::idx_t* got,
        const int* truth,
        int k,
        int truthK,
        int canonicalBase) {
    std::set<faiss::idx_t> truthSet;
    for (int i = 0; i < std::min(k, truthK); ++i) {
        truthSet.insert(truth[i]);
    }

    int hit = 0;
    for (int i = 0; i < k; ++i) {
        faiss::idx_t id = canonicalBase > 0 ? got[i] % canonicalBase : got[i];
        if (truthSet.count(id) != 0) {
            ++hit;
        }
    }
    return double(hit) / double(std::min(k, truthK));
}

double overlapBetween(
        const faiss::idx_t* exact,
        const faiss::idx_t* mc,
        int k,
        int canonicalBase) {
    std::set<faiss::idx_t> exactSet;
    for (int i = 0; i < k; ++i) {
        faiss::idx_t id =
                canonicalBase > 0 ? exact[i] % canonicalBase : exact[i];
        exactSet.insert(id);
    }

    int hit = 0;
    for (int i = 0; i < k; ++i) {
        faiss::idx_t id = canonicalBase > 0 ? mc[i] % canonicalBase : mc[i];
        if (exactSet.count(id) != 0) {
            ++hit;
        }
    }
    return double(hit) / double(k);
}

double rowMax(const float* values, int k) {
    float out = values[0];
    for (int i = 1; i < k; ++i) {
        out = std::max(out, values[i]);
    }
    return double(out);
}

double tieAwareScoreValidity(
        const float* mcDistances,
        int k,
        double exactBoundary,
        double eps) {
    int valid = 0;
    for (int i = 0; i < k; ++i) {
        if (double(mcDistances[i]) <= exactBoundary + eps) {
            ++valid;
        }
    }
    return double(valid) / double(k);
}

template <typename T>
double percentile(std::vector<T> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    double pos = (p / 100.0) * double(values.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(pos));
    size_t hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) {
        return double(values[lo]);
    }
    double w = pos - double(lo);
    return double(values[lo]) * (1.0 - w) + double(values[hi]) * w;
}

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto needValue = [&]() -> const char* {
            if (i + 1 >= argc) {
                fail("missing value for " + key);
            }
            return argv[++i];
        };

        if (key == "--data-dir") {
            args.dataDir = needValue();
        } else if (key == "--queries") {
            args.queries = std::atoi(needValue());
        } else if (key == "--batch-size") {
            args.batchSize = std::atoi(needValue());
        } else if (key == "--k") {
            args.k = std::atoi(needValue());
        } else if (key == "--sample-n") {
            args.sampleN = std::atoi(needValue());
        } else if (key == "--sample-k") {
            args.sampleK = std::atoi(needValue());
        } else if (key == "--candidate-cap") {
            args.candidateCap = std::atoi(needValue());
        } else if (key == "--base-repeat") {
            args.baseRepeat = std::atoi(needValue());
        } else if (key == "--warmup") {
            args.warmup = std::atoi(needValue());
        } else if (key == "--repeats") {
            args.repeats = std::atoi(needValue());
        } else if (key == "--seed") {
            args.seed = std::atoi(needValue());
        } else if (key == "--overflow-cutoff") {
            args.overflowCutoff = true;
        } else if (key == "--cutoff-seed") {
            args.cutoffSeed = std::strtoull(needValue(), nullptr, 10);
        } else {
            fail("unknown argument " + key);
        }
    }
    return args;
}

RunStats runOnce(
        faiss::gpu::GpuResources* resources,
        cublasHandle_t handle,
        cudaStream_t stream,
        faiss::gpu::DeviceTensor<float, 2, true>& base,
        faiss::gpu::DeviceTensor<float, 1, true>& baseNorms,
        faiss::gpu::DeviceTensor<float, 2, true>& queries,
        faiss::gpu::DeviceTensor<float, 2, true>& samples,
        faiss::gpu::DeviceTensor<float, 1, true>& sampleNorms,
        const IntMatrix& gt,
        const Args& args,
        int queryCount,
        bool collectQuality) {
    using namespace faiss;
    using namespace faiss::gpu;

    RunStats stats;
    stats.candidateMin = std::numeric_limits<int>::max();

    DeviceTensor<float, 2, true> scores(
            resources,
            makeDevAlloc(AllocType::Other, 0),
            {args.batchSize, base.getSize(0)});
    DeviceTensor<float, 2, true> repeatedScores;
    if (args.baseRepeat > 1) {
        repeatedScores = DeviceTensor<float, 2, true>(
                resources,
                makeDevAlloc(AllocType::Other, 0),
                {args.batchSize, base.getSize(0) * args.baseRepeat});
    }
    DeviceTensor<float, 2, true> sampleScores(
            resources,
            makeDevAlloc(AllocType::Other, 0),
            {args.batchSize, samples.getSize(0)});
    DeviceTensor<float, 2, true> exactDistances(
            resources, makeDevAlloc(AllocType::Other, 0), {args.batchSize, args.k});
    DeviceTensor<idx_t, 2, true> exactIndices(
            resources, makeDevAlloc(AllocType::Other, 0), {args.batchSize, args.k});
    DeviceTensor<float, 2, true> mcDistances(
            resources, makeDevAlloc(AllocType::Other, 0), {args.batchSize, args.k});
    DeviceTensor<idx_t, 2, true> mcIndices(
            resources, makeDevAlloc(AllocType::Other, 0), {args.batchSize, args.k});
    DeviceTensor<int, 1, true> counts(
            resources, makeDevAlloc(AllocType::Other, 0), {args.batchSize});
    DeviceTensor<int, 1, true> boundaryTieCounts(
            resources, makeDevAlloc(AllocType::Other, 0), {args.batchSize});

    for (int start = 0; start < queryCount; start += args.batchSize) {
        DeviceTensor<float, 2, true> queryBatch(
                queries.data() + size_t(start) * queries.getSize(1),
                {args.batchSize, queries.getSize(1)});

        double sampleDistanceS = timeOnce(stream, [&] {
            computeScores(handle, stream, queryBatch, samples, sampleNorms, sampleScores);
        });
        stats.sampleDistanceS += sampleDistanceS;

        double distanceS = timeOnce(stream, [&] {
            computeScores(handle, stream, queryBatch, base, baseNorms, scores);
        });
        stats.distanceS += distanceS;

        Tensor<float, 2, true>* selectScores = &scores;
        double scoreRepeatS = 0.0;
        if (args.baseRepeat > 1) {
            scoreRepeatS = timeOnce(stream, [&] {
                constexpr int threads = 256;
                dim3 blocks(
                        utils::divUp(
                                repeatedScores.getSize(1),
                                faiss::idx_t(threads)),
                        static_cast<unsigned int>(args.batchSize));
                repeatScoresKernel<<<blocks, threads, 0, stream>>>(
                        scores, repeatedScores);
            });
            stats.scoreRepeatS += scoreRepeatS;
            selectScores = &repeatedScores;
        }

        double exactSelectionS = timeOnce(stream, [&] {
            runBlockSelect(
                    *selectScores,
                    exactDistances,
                    exactIndices,
                    false,
                    args.k,
                    stream);
        });
        stats.exactSelectionS += exactSelectionS;

        double mcSelectionS = timeOnce(stream, [&] {
            runMonteCarloTopKFromScores(
                    resources,
                    stream,
                    *selectScores,
                    sampleScores,
                    args.sampleK,
                    args.k,
                    args.candidateCap,
                    false,
                    mcDistances,
                    mcIndices,
                    &counts,
                    args.overflowCutoff,
                    args.cutoffSeed);
        });
        stats.mcSelectionS += mcSelectionS;
        stats.exactTotalBatchS.push_back(
                distanceS + scoreRepeatS + exactSelectionS);
        stats.mcTotalBatchS.push_back(
                sampleDistanceS + distanceS + scoreRepeatS + mcSelectionS);
        stats.mcSelectionBatchS.push_back(mcSelectionS);

        if (collectQuality) {
            constexpr int threads = 256;
            boundaryTieCountKernel<<<
                    static_cast<unsigned int>(args.batchSize),
                    threads,
                    0,
                    stream>>>(
                    *selectScores,
                    exactDistances,
                    boundaryTieCounts,
                    args.k,
                    1e-5f);

            HostTensor<idx_t, 2, true> exactHost(exactIndices, stream);
            HostTensor<idx_t, 2, true> mcHost(mcIndices, stream);
            HostTensor<float, 2, true> exactDistanceHost(
                    exactDistances, stream);
            HostTensor<float, 2, true> mcDistanceHost(mcDistances, stream);
            HostTensor<int, 1, true> countHost(counts, stream);
            HostTensor<int, 1, true> tieHost(boundaryTieCounts, stream);

            for (int row = 0; row < args.batchSize; ++row) {
                double recallVsExact = overlapBetween(
                        exactHost[row].data(),
                        mcHost[row].data(),
                        args.k,
                        base.getSize(0));
                stats.recallVsExactSum += recallVsExact;
                stats.recallVsExactMin =
                        std::min(stats.recallVsExactMin, recallVsExact);
                double exactBoundary =
                        rowMax(exactDistanceHost[row].data(), args.k);
                double mcBoundary = rowMax(mcDistanceHost[row].data(), args.k);
                double scoreValid =
                        mcBoundary <= exactBoundary + 1e-5 ? 1.0 : 0.0;
                stats.scoreValidSum += scoreValid;
                stats.scoreValidMin =
                        std::min(stats.scoreValidMin, scoreValid);
                double tieAware = tieAwareScoreValidity(
                        mcDistanceHost[row].data(),
                        args.k,
                        exactBoundary,
                        1e-5);
                stats.tieAwareSum += tieAware;
                stats.tieAwareMin = std::min(stats.tieAwareMin, tieAware);
                stats.exactGtSum += overlapAtK(
                        exactHost[row].data(),
                        gt.data.data() + size_t(start + row) * gt.dim,
                        args.k,
                        gt.dim,
                        base.getSize(0));
                stats.mcGtSum += overlapAtK(
                        mcHost[row].data(),
                        gt.data.data() + size_t(start + row) * gt.dim,
                        args.k,
                        gt.dim,
                        base.getSize(0));

                int count = countHost.data()[row];
                stats.candidateSum += count;
                stats.candidateMin = std::min(stats.candidateMin, count);
                stats.candidateMax = std::max(stats.candidateMax, count);
                stats.candidateCounts.push_back(count);
                int boundaryTies = tieHost.data()[row];
                stats.boundaryTieSum += boundaryTies;
                stats.boundaryTieMax =
                        std::max(stats.boundaryTieMax, boundaryTies);
                if (boundaryTies > 1) {
                    ++stats.boundaryTieRows;
                }
                if (count < args.k) {
                    ++stats.underflows;
                }
                if (count > args.candidateCap) {
                    ++stats.overflows;
                }
                if (count < args.k ||
                    (!args.overflowCutoff && count > args.candidateCap)) {
                    ++stats.fallbacks;
                }
            }
            stats.qualityRows += args.batchSize;
        }
    }

    if (!collectQuality) {
        stats.candidateMin = 0;
    }
    return stats;
}

} // namespace

int main(int argc, char** argv) {
    using namespace faiss;
    using namespace faiss::gpu;

    Args args = parseArgs(argc, argv);
    if (args.batchSize <= 0 || args.k <= 0 || args.sampleN <= 0 ||
        args.sampleK <= 0 || args.sampleK > args.sampleN ||
        args.candidateCap < args.k || args.baseRepeat <= 0 ||
        args.repeats <= 0) {
        fail("invalid benchmark arguments");
    }

    std::cout << "loading SIFT1M from " << args.dataDir << std::endl;
    Matrix base = readFvecs(joinPath(args.dataDir, "sift_base.fvecs"));
    Matrix queries = readFvecs(joinPath(args.dataDir, "sift_query.fvecs"));
    IntMatrix gt = readIvecs(joinPath(args.dataDir, "sift_groundtruth.ivecs"));

    int queryCount = std::min(args.queries, queries.rows);
    queryCount = (queryCount / args.batchSize) * args.batchSize;
    if (queryCount <= 0) {
        fail("queries must include at least one full batch");
    }
    if (args.sampleN > base.rows) {
        fail("sample-n exceeds base rows");
    }
    if (args.k > base.rows * args.baseRepeat) {
        fail("k exceeds base rows");
    }
    if (gt.rows < queryCount) {
        fail("groundtruth has fewer rows than requested queries");
    }

    std::vector<float> baseNorm = rowNorms(base);
    auto ids = sampleIds(base.rows, args.sampleN, args.seed);
    Matrix samples = gatherRows(base, ids);
    std::vector<float> sampleNorm = rowNorms(samples);

    StandardGpuResources res;
    auto resources = res.getResources().get();
    auto stream = resources->getDefaultStreamCurrentDevice();

    cublasHandle_t handle;
    cublasCheck(cublasCreate(&handle), "cublasCreate");
    cublasCheck(cublasSetStream(handle, stream), "cublasSetStream");

    HostTensor<float, 2, true> hostBase =
            hostTensorFromVector<float, 2>(base.data, {base.rows, base.dim});
    HostTensor<float, 2, true> hostQueries = hostTensorFromVector<float, 2>(
            queries.data, {queries.rows, queries.dim});
    HostTensor<float, 2, true> hostSamples = hostTensorFromVector<float, 2>(
            samples.data, {samples.rows, samples.dim});
    HostTensor<float, 1, true> hostBaseNorm =
            hostTensorFromVector<float, 1>(baseNorm, {base.rows});
    HostTensor<float, 1, true> hostSampleNorm =
            hostTensorFromVector<float, 1>(sampleNorm, {samples.rows});

    DeviceTensor<float, 2, true> gpuBase(
            resources, makeDevAlloc(AllocType::Other, stream), hostBase);
    DeviceTensor<float, 2, true> gpuQueries(
            resources, makeDevAlloc(AllocType::Other, stream), hostQueries);
    DeviceTensor<float, 2, true> gpuSamples(
            resources, makeDevAlloc(AllocType::Other, stream), hostSamples);
    DeviceTensor<float, 1, true> gpuBaseNorm(
            resources, makeDevAlloc(AllocType::Other, stream), hostBaseNorm);
    DeviceTensor<float, 1, true> gpuSampleNorm(
            resources, makeDevAlloc(AllocType::Other, stream), hostSampleNorm);
    CUDA_VERIFY(cudaStreamSynchronize(stream));

    cudaDeviceProp prop;
    int device = 0;
    CUDA_VERIFY(cudaGetDevice(&device));
    CUDA_VERIFY(cudaGetDeviceProperties(&prop, device));

    std::cout << "device=" << prop.name << " base=" << base.rows
              << " effective_base=" << base.rows * args.baseRepeat
              << " query=" << queryCount << " dim=" << base.dim
              << " k=" << args.k << " batch=" << args.batchSize
              << " sample_n=" << args.sampleN
              << " sample_k=" << args.sampleK
              << " cap=" << args.candidateCap
              << " base_repeat=" << args.baseRepeat
              << " overflow_cutoff=" << (args.overflowCutoff ? 1 : 0)
              << " cutoff_seed=" << args.cutoffSeed << std::endl;

    for (int i = 0; i < args.warmup; ++i) {
        (void)runOnce(
                resources,
                handle,
                stream,
                gpuBase,
                gpuBaseNorm,
                gpuQueries,
                gpuSamples,
                gpuSampleNorm,
                gt,
                args,
                queryCount,
                false);
    }

    std::vector<RunStats> runs;
    for (int i = 0; i < args.repeats; ++i) {
        bool collect = i == args.repeats - 1;
        RunStats stats = runOnce(
                resources,
                handle,
                stream,
                gpuBase,
                gpuBaseNorm,
                gpuQueries,
                gpuSamples,
                gpuSampleNorm,
                gt,
                args,
                queryCount,
                collect);
        runs.push_back(stats);

        double exactTotal =
                stats.distanceS + stats.scoreRepeatS + stats.exactSelectionS;
        double mcTotal = stats.sampleDistanceS + stats.distanceS +
                stats.scoreRepeatS + stats.mcSelectionS;
        double exactPhysical =
                stats.distanceS * args.baseRepeat + stats.exactSelectionS;
        double mcPhysical = stats.sampleDistanceS +
                stats.distanceS * args.baseRepeat + stats.mcSelectionS;
        std::cout << "repeat=" << i << std::fixed << std::setprecision(6)
                  << " distance_s=" << stats.distanceS
                  << " sample_distance_s=" << stats.sampleDistanceS
                  << " score_repeat_s=" << stats.scoreRepeatS
                  << " exact_selection_s=" << stats.exactSelectionS
                  << " mc_selection_s=" << stats.mcSelectionS
                  << " selection_speedup="
                  << stats.exactSelectionS / stats.mcSelectionS
                  << " exact_total_s=" << exactTotal
                  << " mc_total_s=" << mcTotal
                  << " total_speedup=" << exactTotal / mcTotal
                  << " exact_physical_estimate_s=" << exactPhysical
                  << " mc_physical_estimate_s=" << mcPhysical
                      << " physical_estimate_speedup="
                      << exactPhysical / mcPhysical;
        if (collect) {
            std::cout << " recall_vs_exact="
                      << stats.recallVsExactSum / stats.qualityRows
                      << " recall_vs_exact_min=" << stats.recallVsExactMin
                      << " score_valid_mean="
                      << stats.scoreValidSum / stats.qualityRows
                      << " score_valid_min=" << stats.scoreValidMin
                      << " tie_aware_mean="
                      << stats.tieAwareSum / stats.qualityRows
                      << " tie_aware_min=" << stats.tieAwareMin
                      << " exact_gt=" << stats.exactGtSum / stats.qualityRows
                      << " mc_gt=" << stats.mcGtSum / stats.qualityRows
                      << " candidate_mean="
                      << double(stats.candidateSum) / stats.qualityRows
                      << " candidate_p50="
                      << percentile(stats.candidateCounts, 50.0)
                      << " candidate_p95="
                      << percentile(stats.candidateCounts, 95.0)
                      << " candidate_p99="
                      << percentile(stats.candidateCounts, 99.0)
                      << " candidate_min=" << stats.candidateMin
                      << " candidate_max=" << stats.candidateMax
                      << " boundary_tie_mean="
                      << double(stats.boundaryTieSum) / stats.qualityRows
                      << " boundary_tie_rows=" << stats.boundaryTieRows
                      << " boundary_tie_max=" << stats.boundaryTieMax
                      << " underflows=" << stats.underflows
                      << " overflows=" << stats.overflows
                      << " fallbacks=" << stats.fallbacks;
        }
        std::cout << std::endl;
    }

    RunStats mean;
    for (const auto& stats : runs) {
        mean.distanceS += stats.distanceS;
        mean.sampleDistanceS += stats.sampleDistanceS;
        mean.scoreRepeatS += stats.scoreRepeatS;
        mean.exactSelectionS += stats.exactSelectionS;
        mean.mcSelectionS += stats.mcSelectionS;
    }
    double denom = double(runs.size());
    mean.distanceS /= denom;
    mean.sampleDistanceS /= denom;
    mean.scoreRepeatS /= denom;
    mean.exactSelectionS /= denom;
    mean.mcSelectionS /= denom;
    double exactTotal = mean.distanceS + mean.scoreRepeatS + mean.exactSelectionS;
    double mcTotal = mean.sampleDistanceS + mean.distanceS + mean.scoreRepeatS +
            mean.mcSelectionS;
    double exactPhysical = mean.distanceS * args.baseRepeat + mean.exactSelectionS;
    double mcPhysical = mean.sampleDistanceS + mean.distanceS * args.baseRepeat +
            mean.mcSelectionS;
    const auto& last = runs.back();

    std::cout << "summary " << std::fixed << std::setprecision(6)
              << "mean_distance_s=" << mean.distanceS
              << " mean_sample_distance_s=" << mean.sampleDistanceS
              << " mean_score_repeat_s=" << mean.scoreRepeatS
              << " mean_exact_selection_s=" << mean.exactSelectionS
              << " mean_mc_selection_s=" << mean.mcSelectionS
              << " selection_speedup=" << mean.exactSelectionS / mean.mcSelectionS
              << " exact_total_s=" << exactTotal
              << " mc_total_s=" << mcTotal
              << " total_speedup=" << exactTotal / mcTotal
              << " exact_physical_estimate_s=" << exactPhysical
              << " mc_physical_estimate_s=" << mcPhysical
              << " physical_estimate_speedup=" << exactPhysical / mcPhysical
              << " recall_vs_exact=" << last.recallVsExactSum / last.qualityRows
              << " recall_vs_exact_min=" << last.recallVsExactMin
              << " score_valid_mean=" << last.scoreValidSum / last.qualityRows
              << " score_valid_min=" << last.scoreValidMin
              << " tie_aware_mean=" << last.tieAwareSum / last.qualityRows
              << " tie_aware_min=" << last.tieAwareMin
              << " exact_gt=" << last.exactGtSum / last.qualityRows
              << " mc_gt=" << last.mcGtSum / last.qualityRows
              << " candidate_mean="
              << double(last.candidateSum) / last.qualityRows
              << " candidate_p50=" << percentile(last.candidateCounts, 50.0)
              << " candidate_p95=" << percentile(last.candidateCounts, 95.0)
              << " candidate_p99=" << percentile(last.candidateCounts, 99.0)
              << " candidate_min=" << last.candidateMin
              << " candidate_max=" << last.candidateMax
              << " boundary_tie_mean="
              << double(last.boundaryTieSum) / last.qualityRows
              << " boundary_tie_rows=" << last.boundaryTieRows
              << " boundary_tie_max=" << last.boundaryTieMax
              << " exact_batch_p50_s="
              << percentile(last.exactTotalBatchS, 50.0)
              << " exact_batch_p95_s="
              << percentile(last.exactTotalBatchS, 95.0)
              << " exact_batch_p99_s="
              << percentile(last.exactTotalBatchS, 99.0)
              << " mc_batch_p50_s=" << percentile(last.mcTotalBatchS, 50.0)
              << " mc_batch_p95_s=" << percentile(last.mcTotalBatchS, 95.0)
              << " mc_batch_p99_s=" << percentile(last.mcTotalBatchS, 99.0)
              << " mc_select_batch_p50_s="
              << percentile(last.mcSelectionBatchS, 50.0)
              << " mc_select_batch_p95_s="
              << percentile(last.mcSelectionBatchS, 95.0)
              << " mc_select_batch_p99_s="
              << percentile(last.mcSelectionBatchS, 99.0)
              << " underflows=" << last.underflows
              << " overflows=" << last.overflows
              << " fallbacks=" << last.fallbacks << std::endl;

    cublasCheck(cublasDestroy(handle), "cublasDestroy");
    return 0;
}
