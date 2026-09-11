# AGENTS.md

Working notes for the PDX repo.

## What it is

Fast IVF-based similarity search for high-dim vector embeddings, on `float32` **or quantized** vectors (`sq8`) — you pass `float32`, it quantizes internally. Faster than FAISS at equal quality (~1M×1536 in seconds). Header-only **C++17** + **Python bindings**; CPUs (ARM + x86). Paper: https://arxiv.org/html/2503.04422.

## Core idea: Efficient memory layout

PDX is a data layout that transposes vectors in a column-major order. This layout unleashes the true potential of dimension pruning. Pruning means avoiding checking all the dimensions of a vector to determine if it is a neighbour of a query, accelerating index construction and similarity search by factors. Our main pruner is based on ADSampling (paper: https://dl.acm.org/doi/pdf/10.1145/3589282), which is almost-lossless (< 0.005 recall loss). 

Docs: `README.md`, `INSTALL.md`, `BENCHMARKING.md`, `CONTRIBUTING.md`, `examples/README.md`.

### The data layout

PDX is a transposed layout (a.k.a. columnar, or decomposed layout), meaning that the dimensions of different vectors are stored sequentially. This decomposition occurs within a block (e.g., a cluster in an IVF index). The layout evolved from the one presented in the initial paper to reduce random access, and adapted it to work with 8-bit.

#### float32
For float32, the first 25% of the dimensions are fully decomposed. We refer to this as the "vertical block." The rest (75%) are decomposed into subvectors of 64 dimensions. We refer to this as the "horizontal block." The vertical block is used for efficient pruning, and the horizontal block is accessed on the candidates that were not pruned. This horizontal block is still decomposed every 64 dimensions. The idea behind this is that we still have a chance to prune the few remaining candidates every 64 dimensions.

The split is computed by `GetPDXDimensionSplit()` in `include/pdx/common.hpp`: the 25/75 rule holds for `d > 128`; for `d <= 128` the proportions flip (75% vertical), the horizontal block is rounded to a multiple of `H_DIM_SIZE = 64`, and for `d <= 64` everything is vertical. The `static_assert`s next to it are the reference table.

#### 8 bits
Smaller data types are not friendly to PDX, as we must accumulate distances on wider types, resulting in asymmetry. We can work around this by changing the PDX layout. For 8 bits, the vertical block is decomposed every 4 dimensions. This allows us to use dot-product instructions (VPDPBUSD on x86 and UDOT/SDOT on NEON) to calculate L2 or IP kernels while still benefiting from PDX. The horizontal block remains decomposed every 64 dimensions.


## Index types:
All in `include/pdx/indexes/`, templated on `Quantization` (`F32`/`U8`) and sharing `IPDXIndex`
(`ivf_vanilla.hpp`); the search loop itself is `PDXearch` in `searcher.hpp`.
- Vanilla IVF: Plain $k$-means partitioned centroids — `PDXIndex` (`ivf_vanilla.hpp`, storage `IVF` in
  `ivf_core.hpp`). Python: `IndexPDXIVF` / `IndexPDXIVFSQ8`.
- Tree IVF: A layer of mesoclusters is added on top of the plain IVF centroids, where PDX-pruning is also
  applied — `PDXTreeIndex` (`ivf_tree.hpp`, storage `IVFTree`). Python: `IndexPDXIVFTree` /
  `IndexPDXIVFTreeSQ8` (the fastest, README's default).
- Forest IVF (experimental): A forest of many IVF trees. Data is partitioned horizontally, and each
  partition contains an IVF index — `PDXForestIndex` (`ivf_forest.hpp`). Not in `PDXIndexType`, not in
  the Python bindings, not covered by tests; benchmarked via `BenchmarkForest*`.

Serialization / benchmark ids follow `PDXIndexType` in `common.hpp` (`pdx_f32`, `pdx_u8`, `pdx_tree_f32`,
`pdx_tree_u8`). Tree indexes are currently **skipped** in `test_serialization.cpp`,
`test_filtered_search.cpp` and `generate_test_ground_truth.cpp` ("once tree index crash is fixed").

## Resumable search (cursor)

`PDXearch<Q>::IterativeSearch<FILTERED>` (from `BeginIterativeSearch` / `BeginFilteredIterativeSearch`,
or type-erased as `IIterativeSearch` via `IPDXIndex::BeginIterativeSearch(query, k, heap, mutex,
passing_row_ids*)`) owns all per-query state, so any number of cursors run concurrently on one searcher.
- `Next(n)` probes the next n of the `queued_clusters`, ranked once at `Begin`; empty clusters and,
  when filtered, clusters without passing tuples are not queued. `Done()` ⇔ the queue is exhausted. It
  never looks at the heap: callers stop on "heap holds k entries **or** every cursor is done".
- The `TopKHeap` (`heap`, `mutex`, `thread_safe`) belongs to the caller; construct it thread-safe when
  several cursors share it and they take `GetLock()` on every threshold read and merge. While the heap
  holds fewer than k entries a cursor runs `Start`/`FilteredStart` under the lock, exactly like the
  single-shot loop, and `GetPruningThreshold` returns the mask value (no real pruning) as a guard
  against an empty heap.
- Single-shot `Search`/`FilteredSearch` are unchanged and bit-identical to a cursor drained in any chunk
  size (`tests/test_iterative_search.cpp`). The cursor does not bump `n_accessed`.
- `BenchmarkIterativeFiltered <dataset> [index_type] [nprobe] [selectivity]` mirrors the DuckDB
  extension's loop: `Next(nprobe)`, then `Next(5)` until k results or `Done()`.

## Maintenance (SPFresh-like appends/deletes)

Every index implements `Append(row_id, embedding)` / `Delete(row_id)` (pure virtual on `IPDXIndex`;
Python `append`/`delete`). `PDXIndex` and `PDXTreeIndex` follow the same recipe; the tree additionally
keeps the meso-cluster layer (L0) in sync. The leaf-level helpers they share live in
`indexes/ivf_utils.hpp` (`QuantizeAndAppend`, `DequantizeClusterEmbeddings`, `PartitionClusterForSplit`,
`StealNeighborEmbeddings`, `ComputeCentroidMean`, `CalculateDistanceFromEmbeddingToCluster`, ...).
- **Append**: normalize+rotate → nearest centroid (vanilla: exact scan of all centroids; tree: PDX
  search over L0) → `Cluster::AppendEmbedding` (quantized for `U8`) → `row_id_cluster_mapping` →
  `CheckClusterHealth`. Centroids never move on a plain append.
- **Delete**: tombstone the slot (`DeleteEmbedding`), mark the mapping `DELETED_MARKER`,
  `CheckClusterHealth`. Search masks tombstones; `Save()` compacts them away.
- **CheckClusterHealth** runs after *every* mutation: a full cluster (`used_capacity ==
  max_capacity`) is compacted if it has tombstones, else split; `num_embeddings <= min_capacity`
  destroys + merges it (vanilla skips the merge when a single cluster is left).
- **SplitCluster**: 2-means (`SPLIT_KMEANS_ITERS`); points closer to one of the
  `SPLIT_MAX_NEIGHBOR_CLUSTERS` nearest clusters go to "rest" and get reassigned; neighbors' points
  closer to A/B than to their own centroid are stolen; A replaces the old slot, B is `push_back`ed.
  Neighbor set: tree = siblings in the meso-cluster, vanilla = nearest centroids overall.
- **DestroyAndMergeCluster**: swap-and-pop the dead cluster (fix `id`, centroid and mapping of the
  moved one), then `ReassignEmbeddings` (nearest centroid via a `skmeans::BatchComputer` GEMM) with
  merges disabled to avoid cascades.
- **Invariants**: `ReserveClusterSlotIfNeeded()` before holding a `cluster_t&` (splits `push_back`);
  every structural change ends with `ComputeClusterOffsets()` (the searcher sizes its buffers from
  `max_cluster_capacity` on each query); single writer thread. After `Restore()` the truth is
  `index.is_normalized` and `searcher->quantizer` — `config` is only partially recovered (no
  seed/metric on disk), so don't add code paths that depend on it.
- Capacity knobs: `indexes/cluster.hpp` (`CAPACITY_THRESHOLD`, `MIN_CAPACITY_THRESHOLD`,
  `MIN_MAX_CAPACITY = 256`, so small clusters need 256 slots before they split). Split knobs:
  `common.hpp`.
- Tests: `tests/test_maintenance.cpp` (all 4 index types, incl. forced split/merge and
  after-restore). Benchmarks: `BenchmarkInsertion <dataset> [index_type] [nprobe] [build_fraction]`
  and `BenchmarkWorkload <dataset> [index_type] [nprobe]` (edit `WORKLOAD` in `pdx_workload.cpp`).

## Verification gate (definition of done)

**When you believe a feature is finished, prompt the user to run the verification gate. DO NOT RUN WITH EACH CODE CHANGE YOU MAKE. Run in the FOREGROUND — never background these.**

1. **Format** — `./scripts/format.sh`, then `./scripts/format_check.sh` clean.
2. **Build** — `cmake . -DPDX_COMPILE_TESTS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && make -j$(nproc) tests`,
   no errors (the compile-commands export is what step 4 reads).
3. **C++ tests** — `ctest --output-on-failure` all pass.
4. **Lint** — `./scripts/tidy_check.sh`: no `.clang-tidy` warnings from `include/pdx/`.
5. **Python** — `venv/bin/pip install .` (builds the bindings).
6. **Examples** — Run all Python examples inside `./examples` (`venv/bin/python3 examples/<name>.py`).
   Only `pdx_simple.py` is self-contained; the rest read `.hdf5` datasets from
   `benchmarks/datasets/downloaded/` (see `examples/README.md`) — skip those if the data isn't at hand.
7. **AGENTS.md accurate** — if the change invalidated anything here (paths, commands, contracts,
   thresholds, gotchas, SIMD org, style), update this file in the same change.

New feature ⇒ ship a unit test with it (C++ in `tests/`, Python in `python/tests/` if exposed).

## Build & run (beyond the gate)

Header-only; consumers link the `PDX` INTERFACE target (alias `PDX::PDX`), which carries the
include dirs (`include/`, bundled `extern/Eigen`, `extern/SuperKMeans/include`), BLAS/OpenMP/FFTW
links, compile definitions and the `-march` flags. Benchmark binaries have **no** `.out` suffix.
```bash
cmake . -DPDX_COMPILE_BENCHMARKS=ON && make benchmarks
# Index building + search (index_type defaults to pdx_f32; nprobe 0/omitted sweeps a preset list)
# `index_type`: pdx_f32, pdx_u8, pdx_tree_f32, pdx_tree_u8   (forest has its own BenchmarkForest* binaries)
./benchmarks/BenchmarkEndToEnd <dataset_id> [index_type] [nprobe]
```
Add a benchmark with `pdx_add_benchmark(<Name> <source>)` in `benchmarks/CMakeLists.txt`; a test with
`pdx_add_test(<name>.out <source>)` in `tests/CMakeLists.txt` (+ the `tests` custom target list).

Knobs: `-DPDX_MARCH` (default `native`, empty disables `-march`), `-DPDX_PORTABLE` (`-mavx2 -mfma` on
x86_64 / plain `-O3` elsewhere, for wheels; also via the `PDX_PORTABLE` env var in `pip install .`),
`-DPDX_SKIP_FFTW`, `-DBLAS_LIBRARIES` (a good BLAS is critical — distro/apt OpenBLAS is slow, build
from source). See INSTALL.md.


## Code style

- **Naming**: `PascalCase` functions/classes/structs; `snake_case` variables/members;
  `UPPER_SNAKE_CASE` constants; everything lives in `namespace PDX` (the SuperKMeans dependency
  is `skmeans::`).
- Follow `.clang-format` / `.clang-tidy`.
- **Memory**: RAII, no raw `new`/`delete`. Buffers that don't need zero-init → `new T[]` in a
  `unique_ptr`, **not** `std::vector`/`resize()`.
- Headers are `.hpp` (all under `include/pdx/`). Constants/magic numbers → `include/pdx/common.hpp`
  as `constexpr`.
- Keep comments simple. TODOs: `TODO(@<github_user>, <priority>): <summary>`.
- **Reuse before writing**: check `common.hpp`, `utils.hpp`, `indexes/ivf_utils.hpp`,
  `quantizers/scalar.hpp` first.

## Performance

Performance-critical — weigh every copy/allocation.
- **SIMD is centralized.** Distance kernels exist for **NEON / AVX2 / AVX512 / scalar**,
  dispatched at compile time (`base_computers.hpp`, on `__ARM_NEON`/`__AVX2__`/`__AVX512F__`) and
  tagged by `Quantization` (`F32`/`U8`; `F16`/`BF` exist in the enum but have no kernels). Kernels are
  only specialized for `DistanceMetric::L2SQ` — `IP`/`COSINE` are served by normalizing the data
  (`DistanceMetricRequiresNormalization`) and running L2SQ.
  **All** SIMD lives in `include/pdx/distance_computers/` (`neon_computers.hpp`, `avx2_computers.hpp`,
  `avx512_computers.hpp`, `scalar_computers.hpp`) — don't scatter it elsewhere; keep all backends in
  sync when changing a kernel (on ARM the x86 ones aren't compiled or linted). Cover kernels in
  `tests/test_distance_computers.cpp`.
- **`PDX_VECTORIZE_LOOP`** (`common.hpp`) forces loop autovectorization (esp. FP reductions) — put it
  on its own line right above the `for`, never a raw `#pragma clang loop`. Other macros there: `PDX_RESTRICT`, `PDX_ALWAYS_INLINE`, `PDX_NO_INLINE`,
  `PDX_LIKELY`/`PDX_UNLIKELY`, `PDX_PREFETCH`, `PDX_ENSURE_POSITIVE`.
- **Profiling**: `PDX_PROFILE_SCOPE("name")` (`profiler.hpp`); most benchmarks call
  `Profiler::Get().Print()` at the end (e.g. `BenchmarkEndToEnd`).
