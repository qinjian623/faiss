# Faiss

Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning. Faiss is written in C++ with complete wrappers for Python/numpy. Some of the most useful algorithms are implemented on the GPU. It is developed primarily at Meta's [Fundamental AI Research](https://ai.facebook.com/) group.

## News

See [CHANGELOG.md](CHANGELOG.md) for detailed information about latest features.

## Introduction

Faiss contains several methods for similarity search. It assumes that the instances are represented as vectors and are identified by an integer, and that the vectors can be compared with L2 (Euclidean) distances or dot products. Vectors that are similar to a query vector are those that have the lowest L2 distance or the highest dot product with the query vector. It also supports cosine similarity, since this is a dot product on normalized vectors.

Some of the methods, like those based on binary vectors and compact quantization codes, solely use a compressed representation of the vectors and do not require to keep the original vectors. This generally comes at the cost of a less precise search but these methods can scale to billions of vectors in main memory on a single server. Other methods, like HNSW and NSG add an indexing structure on top of the raw vectors to make searching more efficient.

The GPU implementation can accept input from either CPU or GPU memory. On a server with GPUs, the GPU indexes can be used a drop-in replacement for the CPU indexes (e.g., replace `IndexFlatL2` with `GpuIndexFlatL2`) and copies to/from GPU memory are handled automatically. Results will be faster however if both input and output remain resident on the GPU. Both single and multi-GPU usage is supported.

## Installing

Faiss comes with precompiled libraries for Anaconda in Python, see [faiss-cpu](https://anaconda.org/pytorch/faiss-cpu), [faiss-gpu](https://anaconda.org/pytorch/faiss-gpu) and [faiss-gpu-cuvs](https://anaconda.org/pytorch/faiss-gpu-cuvs). The library is mostly implemented in C++, the only dependency is a [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) implementation. Optional GPU support is provided via CUDA or AMD ROCm, and the Python interface is also optional. The backend GPU implementations of NVIDIA [cuVS](https://github.com/rapidsai/cuvs) can also be enabled optionally. It compiles with cmake. See [INSTALL.md](INSTALL.md) for details.

## How Faiss works

Faiss is built around an index type that stores a set of vectors, and provides a function to search in them with L2 and/or dot product vector comparison. Some index types are simple baselines, such as exact search. Most of the available indexing structures correspond to various trade-offs with respect to

- search time
- search quality
- memory used per index vector
- training time
- adding time
- need for external data for unsupervised training

The optional GPU implementation provides what is likely (as of March 2017) the fastest exact and approximate (compressed-domain) nearest neighbor search implementation for high-dimensional vectors, fastest Lloyd's k-means, and fastest small k-selection algorithm known. [The implementation is detailed here](https://arxiv.org/abs/1702.08734).

## Experimental GPU Monte-Carlo TopK patch

This working tree includes an experimental GPU utility,
`faiss/gpu/utils/MonteCarloTopK.cuh`, for studying post-score top-k selection.
It takes a materialized score matrix plus a smaller sample score matrix,
estimates a per-query threshold from the sample, filters the full score row into
a fixed-width candidate buffer, and then runs the existing FAISS GPU
`runBlockSelect` selector on the candidates.

The implementation is intended for benchmarking and research comparison. It
does not replace FAISS indexes and does not reduce flat-search distance
computation. It only changes the selection stage after scores have already been
computed.

### Semantics

`runMonteCarloTopKFromScores` has two modes:

- Strict mode, the default: if a row has fewer than `k` candidates or more than
  `candidateCap` threshold-passing candidates, that row falls back to exact
  full-row top-k. This preserves exact top-k results, up to tie ordering, but
  fallback rows can dominate runtime.
- Overflow-cutoff mode: if `overflowCutoff=true`, rows with more than
  `candidateCap` candidates keep a pseudo-random subset of size `candidateCap`
  and do not fall back for overflow. Rows with fewer than `k` candidates still
  fall back, because they cannot produce `k` valid outputs. This mode is
  approximate and should always be reported with recall versus exact top-k.

The SIFT1M benchmark prints both mean and minimum `recall_vs_exact`, plus
`underflows`, `overflows`, and `fallbacks`, so strict and cutoff results can be
compared without mixing their correctness contracts.

### Build targets

Configure FAISS with GPU tests enabled, then build the MC TopK targets:

```bash
cmake -B build-mc-topk \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-mc-topk --target \
  TestGpuMonteCarloTopK \
  bench_gpu_monte_carlo_topk \
  bench_gpu_sift1m_monte_carlo_topk \
  -j
```

The SIFT1M executable is CUDA-only because it uses cuBLAS to materialize the
flat squared-L2 score matrix.

Run the unit test:

```bash
./build-mc-topk/faiss/gpu/test/TestGpuMonteCarloTopK
```

### SIFT1M benchmark

The benchmark expects TEXMEX SIFT1M files in one directory:

- `sift_base.fvecs`
- `sift_query.fvecs`
- `sift_groundtruth.ivecs`

Strict exact-contract run:

```bash
./build-mc-topk/faiss/gpu/test/bench_gpu_sift1m_monte_carlo_topk \
  --data-dir /path/to/sift1M \
  --queries 9984 \
  --batch-size 32 \
  --k 64 \
  --sample-n 8192 \
  --sample-k 8 \
  --candidate-cap 50000 \
  --warmup 1 \
  --repeats 3 \
  --seed 0
```

Overflow-cutoff run:

```bash
./build-mc-topk/faiss/gpu/test/bench_gpu_sift1m_monte_carlo_topk \
  --data-dir /path/to/sift1M \
  --queries 1024 \
  --batch-size 32 \
  --k 64 \
  --sample-n 8192 \
  --sample-k 4 \
  --candidate-cap 8192 \
  --base-repeat 10 \
  --overflow-cutoff \
  --cutoff-seed 123 \
  --warmup 1 \
  --repeats 3 \
  --seed 0
```

`--base-repeat` virtually repeats the SIFT1M score matrix after distance
computation to stress the top-k selector at larger effective `N`. The benchmark
therefore reports both measured score-matrix speedup and a
`physical_estimate_speedup` that scales the distance term by the repeat factor.

Useful output fields:

- `mean_exact_selection_s`, `mean_mc_selection_s`: selector-only timing.
- `exact_total_s`, `mc_total_s`: measured score-matrix pipeline timing.
- `physical_estimate_speedup`: end-to-end estimate for repeated-score runs.
- `recall_vs_exact`, `recall_vs_exact_min`: mean and worst-row overlap with
  exact top-k from the same score matrix.
- `candidate_mean`, `candidate_min`, `candidate_max`: threshold-passing
  candidate statistics before cap truncation.
- `underflows`, `overflows`, `fallbacks`: candidate safety counters. In cutoff
  mode, `overflows` can be nonzero while `fallbacks` remains zero.

### Reference RTX 4090 results

On an NVIDIA RTX 4090 with CUDA 12.4, SIFT1M, `k=64`, `batch=32`, and three
measured repeats:

| Mode | Effective N | sample_n/sample_k/cap | MC total | Speedup | Selector speedup | Recall vs exact mean/min | Fallbacks |
|---|---:|---|---:|---:|---:|---:|---:|
| Strict | 1M | 8192/4/50000 | 0.0377 s | 2.72x | 10.22x | 0.999893/0.984375 | 0 |
| Strict | 1M, 9984 queries | 8192/8/50000 | 0.3679 s | 2.71x | 9.99x | 0.999804/0.984375 | 0 |
| Strict | 1M, 9984 queries | 8192/4/50000 | 6.0290 s | 0.165x | 0.122x | 0.999804/0.984375 | 18 |
| Cutoff | 10M virtual | 8192/4/8192 | 0.1231 s | 2.91x physical est. | 15.46x | 0.990082/0.531250 | 0 |

The third strict row shows the main failure mode: a small number of fallback
rows can erase the selector-speedup benefit. The cutoff row shows the opposite
tradeoff: overflow fallback is removed, but the result is approximate and the
minimum exact overlap can drop sharply.

## Full documentation of Faiss

The following are entry points for documentation:

- the full documentation can be found on the [wiki page](https://github.com/facebookresearch/faiss/wiki), including a [tutorial](https://github.com/facebookresearch/faiss/wiki/Getting-started), a [FAQ](https://github.com/facebookresearch/faiss/wiki/FAQ) and a [troubleshooting section](https://github.com/facebookresearch/faiss/wiki/Troubleshooting)
- the [doxygen documentation](https://faiss.ai/) gives per-class information extracted from code comments
- to reproduce results from our research papers, [Polysemous codes](https://arxiv.org/abs/1609.01882) and [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734), refer to the [benchmarks README](benchs/README.md). For [
Link and code: Fast indexing with graphs and compact regression codes](https://arxiv.org/abs/1804.09996), see the [link_and_code README](benchs/link_and_code)

## Authors

The main authors of Faiss are:
- [Hervé Jégou](https://github.com/jegou) initiated the Faiss project and wrote its first implementation
- [Matthijs Douze](https://github.com/mdouze) implemented most of the CPU Faiss
- [Jeff Johnson](https://github.com/wickedfoo) implemented all of the GPU Faiss
- [Lucas Hosseini](https://github.com/beauby) implemented the binary indexes and the build system
- [Chengqi Deng](https://github.com/KinglittleQ) implemented NSG, NNdescent and much of the additive quantization code.
- [Alexandr Guzhva](https://github.com/alexanderguzhva) many optimizations: SIMD, memory allocation and layout, fast decoding kernels for vector codecs, etc.
- [Gergely Szilvasy](https://github.com/algoriddle) build system, benchmarking framework.

## Reference

References to cite when you use Faiss in a research paper:
```
@article{douze2024faiss,
      title={The Faiss library},
      author={Matthijs Douze and Alexandr Guzhva and Chengqi Deng and Jeff Johnson and Gergely Szilvasy and Pierre-Emmanuel Mazaré and Maria Lomeli and Lucas Hosseini and Hervé Jégou},
      year={2024},
      eprint={2401.08281},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
For the GPU version of Faiss, please cite:
```
@article{johnson2019billion,
  title={Billion-scale similarity search with {GPUs}},
  author={Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  journal={IEEE Transactions on Big Data},
  volume={7},
  number={3},
  pages={535--547},
  year={2019},
  publisher={IEEE}
}
```

## Join the Faiss community

For public discussion of Faiss or for questions, visit https://github.com/facebookresearch/faiss/discussions.

We monitor the [issues page](https://github.com/facebookresearch/faiss/issues) of the repository.
You can report bugs, ask questions, etc.

## Legal

Faiss is MIT-licensed, refer to the [LICENSE file](https://github.com/facebookresearch/faiss/blob/main/LICENSE) in the top level directory.

Copyright © Meta Platforms, Inc.
