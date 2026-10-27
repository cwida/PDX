# AGENTS.md

Working notes for the PDX repo.

## What it is

Fast IVF-based similarity search for high-dim vector embeddings, on `float32` **or quantized** vectors (`sq8`, or SAQ at 3/4 bits per dimension) — you pass `float32`, it quantizes internally. Faster than FAISS at equal quality (~1M×1536 in seconds). Header-only **C++17** + **Python bindings**; CPUs (ARM + x86). Paper: https://arxiv.org/html/2503.04422.

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

#### SAQ
SAQ (`Quantization::SAQ`, `quantizers/saq.hpp`; arXiv 2509.12086) codes the residual to one global mean in PCA space. A
DP plan cuts the dimensions at 64-dim cuts into segments of widths 1–8, within a 3- or 4-bit budget that counts the fp16
factors and a partial last chunk's full 64-dim record, and each segment gets its own random rotation: the segment is the unit of quantization, the 64-dim chunk the
unit of scan. The plan depends on d: below `SAQ_DP_PLAN_MIN_DIMENSIONS` (256) one segment at the declared bits, its
factors on top of the budget (vanilla RaBitQ; the DP's segments lost 0.11–0.37 recall to it there); from 256 the DP;
from `SAQ_PDX_U8_HEAD_MIN_DIMENSIONS` (640) the first chunk (the head) is forced to its own 8-bit segment in the PDX U8
layout, which costs no recall from 640 on but 0.03 at 384 and 0.25 at 200 (4 bits). A head chunk at any other width is
stored and scanned as a record (`SAQModel::HasPDXU8Head`, `ClusterFormat<SAQ>::GetFirstRecordChunk`). Encoded row: each chunk as a
T2 record (`PackRecord`/`UnpackRecord`: at 8 bits the levels as they are, else the width split into 4-, 2- and 1-bit
parts, each interleaved across byte lanes, 8·w bytes); one fp16 rescale per segment; fp16 |r| (not |r|², which fp16
holds only up to |r| = 256; readers square it). A row with |r| above 65,504 (L2 without normalization only) gets an
infinite estimate and is never returned. In a cluster every part
is transposed at the cluster's stride: the head in the U8 vertical layout if 8-bit, then each chunk's records, then the
factors (illustration above `SAQModel`).


## Index types:
All in `include/pdx/indexes/`, templated on `Quantization` (`F32`/`U8`, plus `SAQ` for the vanilla IVF) and sharing
`IPDXIndex` (`ivf_vanilla.hpp`); the search loop itself is `PDXearch` in `ivf_searcher.hpp`.
- Vanilla IVF: Plain $k$-means partitioned centroids — `PDXIndex` (`ivf_vanilla.hpp`, storage `IVF` in
  `ivf_core.hpp`). Python: `IndexPDXIVF` / `IndexPDXIVFSQ8` / `IndexPDXIVFSAQ` (`bits` 3 or 4).
- Tree IVF: A layer of mesoclusters is added on top of the plain IVF centroids, where PDX-pruning is also
  applied — `PDXTreeIndex` (`ivf_tree.hpp`, storage `IVFTree`). Python: `IndexPDXIVFTree` /
  `IndexPDXIVFTreeSQ8` (the fastest, README's default). `static_assert(Q != SAQ)` (TODO: SAQ in the tree).
- Flat: exact search over one row-major block of transformed `float32` embeddings, for sets too
  small to cluster — `FlatIndex` (`flat.hpp`, storage `Flat` in `flat_core.hpp`, search
  `FlatSearcher` in `flat_searcher.hpp`). Same `IPDXIndex` API (one cluster; the cursor is `Done()`
  after its first `Next`), `PDX_FLAT` in `PDXIndexType`, not in the Python bindings.
  `GetRowIds()`/`GetEmbeddings()` feed a `PDXIndex::BuildIndex` with `is_data_transformed`.
  `IPDXIndex::BulkSearch(queries, n, k, is_query_transformed, executor)`, implemented by Flat only
  (the other indexes throw `std::logic_error`): one transform, then
  SuperKMeans' `BatchComputer::FindKNearestNeighbors` on the given executor, else the bound one: GEMMs
  over 256-query blocks in parallel (one block runs on the calling thread). With at least
  `FLAT_SEARCHER_BULK_SEARCH_SIZE_THRESHOLD` (512) rows and fewer blocks than workers, each worker
  searches a slice of the rows instead and the slices' top-k merge per query. Tombstones get an
  infinite norm. Unfiltered.

Centroids (`Centroids`, `indexes/centroids.hpp`) are stored once, in `PDXIndexConfig::centroid_quantization`: `F32`
(default), `LVQ8` or `LVQ4` (`LVQQuantizer<BIT_WIDTH>`, `quantizers/lvq.hpp`: d 8-bit codes, or two 4-bit codes per
byte with dimension 2i in the low nibble, plus `LVQParams` per centroid; no float32 copy kept). A new encoding adds its
case to `Centroids`' switches (`-Wswitch` lists them) and to `TestUtils::QUANTIZED_CENTROIDS`. Vanilla and tree leaf centroids; the tree's L0 (its centroids and its PDX copies of the leaf centroids)
stays f32, and Flat ignores the setting. Search ranking uses `ComputeApproximateDistancesToCentroids`: LVQ centroids
take the query in int8 (`LVQInt8Query`, scale max|q|/127, through `HorizontalDotInt8WithU8`/`U4`), F32 ones are exact.
`Append`'s nearest centroid and the split's neighbour search keep the exact `ComputeDistancesToCentroids` (LVQ: the L2
to the decoded centroid, through `HorizontalDotWithU8`/`U4`). The int8 kernels read whole
`LVQ_INT8_QUERY_CHUNK_SIZE_IN_BYTES` (64) chunks: the query is zero-padded (so the bytes read past the codes, left
uninitialized, count for nothing), and LVQ centroids keep that many bytes after the last one (`trailing_padding_size`;
F32 none). The encoded query travels as a `CentroidRankingQuery` (built for
one `CentroidQuantization`; a new encoding adds its part to it). LVQ encoding (`MinMax`, `QuantizeToU8`/`U4`) and LVQ4
decoding (`DequantizeFromU4`) are SIMD kernels, with codes bit-identical to scalar; LVQ8 decoding is an auto-vectorized
loop. Maintenance that needs floats decodes into scratch (`GetF32CentroidPtr`, `DecodeCentroid`, and `GetF32CentroidsPtr` +
`GetF32CentroidsSquaredNormsPtr` for the GEMMs). A float centroid compared with decoded ones goes through
`RoundTripThroughEncoding` first, so both sides carry the encoding's error. A centroid is encoded only when computed (build, the split's two
means); merges move encoded centroids as they are.

Serialization / benchmark ids follow `PDXIndexType` in `common.hpp` (`pdx_f32`, `pdx_u8`, `pdx_tree_f32`, `pdx_tree_u8`;
`pdx_saq3`/`pdx_saq4` give `PDX_SAQ` (5) with the config's bits; `PDX_FLAT` is serialization only). `IVF<SAQ>` saves its
`SAQModel`, per-chunk ε and `dirty_calibration` right after the header in all four blocks (`Save`/`Load`, `SaveResidentData`/
`LoadResidentData`); T stays with the pruner, written dense.
- File: `Save(path)` / `LoadPDXIndex(path)` write and read the type byte, the rotation matrix (`WriteRotationMatrix` /
  `ReadRotationMatrix`) and the payload, but no `PDXIndexConfig`: a restored index has the default config (e.g. no
  `max_cluster_size`) but its centroid quantization.
- The type byte (file and stream, `MakeIndexTypeByte`) holds the `PDXIndexType` in its low nibble and the
  `CentroidQuantization` in its high one (F32 = 0, so F32 indexes keep the bytes they had; Flat always writes 0). The
  payload holds the centroids in that encoding; an unknown one throws `std::runtime_error`.
- Stream (for indexes that share one rotation, e.g. one index per partition of a table): `SaveToStream(out)` writes
  `[PDX_SERIALIZATION_VERSION][type][PDXIndexConfig]` (`WriteStreamHeader`) and the payload, no rotation, tombstones
  compacted; `LoadPDXIndexFromStream(in, pruner)` constructs the index on `pruner` (which must hold the saving
  rotation) and calls `LoadFromStream`. A version mismatch throws `std::runtime_error`.
- `PDXIndex`'s stream payload puts first what a search keeps in memory, so a reader can fetch one cluster alone:
  `IVF::SaveResidentData` (cluster sizes and capacities, each cluster's offset in the cluster data, the centroids),
  then the row-id mapping (`RowIdClusterMapping::Save`), then `IVF::SaveClusterData` (per cluster: row ids, then
  compact PDX data). `LoadClusterData` reads by offset and skips gaps and empty clusters (which hold no bytes and share the next cluster's offset), so the writer may reorder or pad clusters.
  `IVFTree` and `FlatIndex` stream the file format's payload.
- Cluster storage order: `SaveResidentData`/`SaveClusterData` lay the clusters out in the order
  `IVFUtils::ComputeClusterStorageOrder` gives for `PDXIndexConfig::cluster_storage_order` (write-time, not
  serialized): `ID`, or `NEAREST_CENTROID_CHAIN` (default), so that the clusters a query probes together share the
  storage blocks a pager reads (DuckIR: 256 KiB). Empty clusters go first, then a greedy nearest-neighbour path over
  the decoded centroids from the first one (serial, k²/2 distances). Above `CLUSTER_STORAGE_ORDER_MAX_CHAIN_CLUSTERS`
  (4096) non-empty clusters: `ComputeKMeans` of the centroids into round(√k) groups, the groups along a path over
  their centroids, and a path inside each group (one `ParallelFor` over the groups) from its member nearest the
  previous group's centroid. The reader needs no order: offsets are stored.
- Paging: `LoadPDXIndexFromStream(in, pruner, &cluster_source)` (`LoadResidentDataFromStream`) loads a `PDXIndex`
  without its clusters' arrays (`data`/`indices` null; Flat and the tree load everything). `ProbeCluster` then gets
  a cluster's bytes from the caller's `IClusterSource` (`Acquire`/`Release`, concurrent; row ids, then PDX data with
  stride `used_capacity`), which locates them with `GetClusterDataRange`. Such an index is read-only but for `Delete`,
  which only tombstones (no `CheckClusterHealth`): the caller replays its deletes on a full load before rewriting it.
  `Append` throws; `GetEmbeddingsFromIndexByRowIds` acquires each cluster it reads once.
- Every format is implemented once: the `Load` functions are templates over a reader (`BufferReader` for buffers,
  `StreamReader` for streams, in `utils.hpp`); `IVF::LoadClusters` reads a level's clusters of the file format for
  `IVF` and `IVFTree`.

## Resumable search (cursor)

`PDXearch<Q>::IterativeSearch<FILTERED>` allows for: i) concurrent queries on one index, ii) resume a search. The API of a resumable search is: `Next(n)`: probes the next n clusters ranked once at `Begin`; `Done()`: the clusters are exhausted. `GetClustersAccessOrder(query)` returns every cluster, nearest first; passing it as `BeginIterativeSearch`'s `clusters_access_order` lets several cursors of one query (e.g. one per batch of passing rows) skip ranking again (Flat returns `{0}`).
`BeginIterativeSearch*`, `GetDistancesToCentroids` and `GetClustersAccessOrder` take an optional `CentroidRankingQuery`
last, so a caller that ranks one query on many indexes (DuckIR: one per row group) encodes it once (null: each index
encodes it). A cursor that ranks every cluster itself (no preset order) computes the distances at `Begin` and orders lazily: `EnsureClustersOrder` partial-sorts the queue as far as `Next` probes, at least doubling the ordered prefix, so it probes in the full sort's order without sorting what it never reaches.

Cursors may share one `thread_safe` `TopKHeap`. Only merges take its lock (`Start`'s full scan of the first clusters runs unlocked), and each merge ends with `PublishKthDistance(k)`; the pruning thresholds read that atomic `kth_distance` (+inf while fewer than k) without the lock, as a stale value is only looser. Every change to `heap` must be followed by `PublishKthDistance(k)` under the lock, and a heap reused for another query is reset with `Clear()`.

For many queries over one filter (e.g. a LATERAL join over a partition): `CreateSharedPredicateEvaluator(passing_row_ids)` builds the filter once and `BeginIterativeSearchWithSharedEvaluator` starts each query's cursor on it (the evaluator must outlive them).

Single-shot `Search`/`FilteredSearch` are thin wrappers: a non-thread-safe `TopKHeap`, one cursor over the n_probe-clamped ranking. Note: The tree's meso-cluster (L0) layer is not supported by cursors or by `FilteredSearch`. Both rank all leaf centroids flat, so a tree cursor is a vanilla IVF search over the tree's leaves.

## Maintenance (SPFresh-like appends/deletes)

Every index implements `Append(row_id, embedding)` / `Delete(row_id)` `PDXTreeIndex` additionally keeps the meso-cluster layer (L0) in sync. The leaf-level helpers they share live in `indexes/ivf_utils.hpp`; operations on one cluster's own data (`StoreEmbeddings`, `Grow`, `CompactCluster`) are `Cluster` members (`indexes/cluster.hpp`).
- **Append**: normalize+rotate → nearest centroid (vanilla: `ComputeDistancesToCentroids` over all centroids; tree: PDX search over L0). Centroids never move on a plain append.
- **BulkAppend** (`row_ids, embeddings, n`): validates every id first (live, repeated in the batch, outside
  [base_row_id, uint32]) and throws before any change. Transforms the batch with one `NormalizeAndRotate`, then runs
  rounds: assign the pending rows with one GEMM (`IVFUtils::AssignToNearestCentroids`; the tree assigns exactly over
  all leaves), group them by cluster (`GroupRowsByCluster`), append in parallel what fits of each cluster's rows while
  one slot stays free (`AppendRowsToClusters` → `Cluster::AppendEmbeddings`), then each overflowing cluster's trigger
  row goes through the `Append` path so the cluster splits at its normal full size (`SplitOverflowedClusters`, merges
  off). Held rows are assigned again next round against the new centroids. Not identical to `Append`: rows are
  placed against the round's centroids. Basic exception guarantee. Flat: one transform, then the `AppendEmbedding` loop.
- **Delete**: tombstone the slot (`DeleteEmbedding`), mark the mapping `DELETED_MARKER`, `CheckClusterHealth`. Search masks tombstones and never lets them into the heap, also while the heap is not yet full (`k` above the live count); `Save()` compacts them away.
- **DestroyAndMergeCluster**: swap-and-pop the dead cluster (fix `id`, centroid and mapping of the moved one), then `ReassignEmbeddings` (nearest centroid via a `skmeans::BatchComputer` GEMM on the decoded floats; the stored encoded rows are appended as they are) with merges disabled to avoid cascades.
- **SplitCluster**: 2-means into A and B; points closer to one of the `SPLIT_MAX_NEIGHBOR_CLUSTERS` nearest centroids go to a "rest" group that is reassigned, and neighbour points closer to A or B are stolen (with `max_cluster_size`, only as many as A and B have room for: `CapRowsToSteal`). If the 2-means leaves A or B empty (identical embeddings cannot be separated), the cluster is not split: `Cluster::Grow()` re-lays it out in `CAPACITY_THRESHOLD` times as many slots (same slots, row ids and centroid). The rest group's reassignment can split other clusters recursively; past `SPLIT_CASCADE_MAX_DEPTH` nested levels a full cluster grows instead of splitting.
- **Size limit**: `Cluster::CapacityFor(n, PDXIndexConfig::max_cluster_size)` sizes the leaf clusters of a build and of a split. With 0 (default) it is `CAPACITY_THRESHOLD` times the creation size (at least 256), and a cluster splits when its slots fill. With a limit, it is `CAPACITY_THRESHOLD` times the rows (at least one free slot), capped at the limit: a full cluster below the limit grows instead (`Cluster::GrowTo`, which moves the used slots into a larger stride and keeps slots and tombstones; `BulkAppend` grows each cluster for its batch first, `Cluster::GrowToFit`), and splits once it holds the limit. `PDXIndexConfig::expect_appends = false` (build-time, not serialized) builds the leaf clusters with one free slot; appends still grow them (with a limit) or split them (without). `IPDXIndex::ShrinkToFit()` leaves one free slot after each leaf cluster's used slots, for an index that stops receiving appends (vanilla also trims the centroids, which splits grow by doubling). Streams save each cluster's capacity, and loads restore it. With a limit, `BuildIndex` also splits the k-means clusters above it (`IVFUtils::SplitClustersAboveMaxClusterSize`: k-means on each cluster's own embeddings into splits of half the limit, repeated until they fit) before populating, so the tree's L0 is built over them. Clusters k-means can't separate (identical embeddings) stay above the limit, at build and on append. L0 clusters and the merge threshold ignore the limit.
- **ReassignEmbeddings** (also used by `SplitCluster` for the "rest" group) always runs with merges disabled: it snapshots the nearest-centroid assignments before its loop, and a merge inside the loop would swap-and-pop cluster ids from under it (see the TODO at its definition for the root fix). Clusters drained by `StealNeighborEmbeddings` therefore merge only when a later Append/Delete touches them.
- **Invariants**: `ReserveClusterSlotIfNeeded()` before holding a `cluster_t&` (splits `push_back`); every structural change, a cluster re-layout included, ends with `ComputeClusterOffsets()` (the searcher sizes its buffers from `max_cluster_capacity` on each query); single writer thread.
- **Row id mapping**: `RowIdClusterMapping` (`ivf_utils.hpp`) is a dense vector indexed by row id holding `(cluster, index_in_cluster)`, `DELETED_MARKER` (`common.hpp`) for absent/deleted ids; exposed as `IPDXIndex::GetRowIdMapping`. Row ids are expected dense (PDX assigns them by position); the vector grows to the largest appended id, counted from `PDXIndexConfig::base_row_id` (for an index that holds one row-id range of a larger table). `Delete` returns whether the id was in the index (an absent id is a no-op), `Contains` whether it is; `Append` of a live id throws.
- **Transformed input**: `PDXIndexConfig::is_data_transformed` makes `BuildIndex`/`Append` take embeddings that are already normalized and rotated. Pair it with the `(config, ADSamplingPruner&)` constructors of `PDXIndex`/`PDXTreeIndex`/`FlatIndex` to share one rotation matrix across indexes; cursors take `is_query_transformed` for the matching query.
- Knobs: `indexes/cluster.hpp` (`CAPACITY_THRESHOLD`, `MIN_CAPACITY_THRESHOLD`, `DEFAULT_MAX_CAPACITY_LOWER_BOUND = 256`, so without a size limit small clusters need 256 slots before they split). Split knobs: `common.hpp` (`SPLIT_MAX_NEIGHBOR_CLUSTERS`, `SPLIT_KMEANS_ITERS`, `SPLIT_CASCADE_MAX_DEPTH`, `MAX_CLUSTER_SIZE_LOWER_BOUND`).

## SAQ index (`PDXIndex<SAQ>`, alias `PDXIndexSAQ`)

- **Fit**: `SAQQuantizer::Fit(rows, n, d, bits, seed, executor, allocator)` (on up to `SAQ_FIT_SAMPLE_SIZE` rows)
  returns the d×d rotation T (PCA, then a random rotation per segment) and the `SAQModel` (plan, rotated mean,
  per-chunk widths, code centres, record offsets and remaining variances, and each segment's `scale`).
- **Encoding**: `EncodeEmbedding` runs `ComputeResidual`, then `EncodeSegment` per segment: the levels of the best
  of `SAQ_SCALE_GRID_POINTS` (4) scales over ±`SAQ_SCALE_GRID_SPAN` (25%) around the segment's `scale`, which the
  fit sets to the mean best scale of `SAQ_CONSTANT_SCALE_VECTORS` (100) Gaussian vectors of the segment's length
  (RaBitQ-Library's `get_const_scaling_factors`), then `PackRecord` per chunk. The exact extended-RaBitQ search
  (`FindBestScale`, a heap over level steps) runs only in the fit, for the scales and the plan's per-width
  distortion: per row it cost 44–799× U8. `DecodeEmbedding` runs `DecodeRecord` per chunk.
- **Standalone build**: `BuildIndex` normalizes once, fits, makes the owned pruner from T (`ADSamplingPruner(d, T, d, …)`:
  dense, no random QR), rotates, runs k-means with `use_superkmeans_pruning = false` (SuperKMeans' bounds need a random
  rotation, T starts with a PCA; also through `SplitClustersAboveMaxClusterSize`), encodes and calibrates the bound.
- **Shared fit (DuckIR)**: `PDXIndex(config, pruner, model)`, usually with `is_data_transformed`. Without a model, an SAQ
  index on an external pruner or on transformed data throws. The loaders read the model from the file or stream.
- **Search**: `ProbeClusterData` runs `SAQStart` (every chunk of every vector, `ComputeSAQEstimates`) while the heap holds
  fewer than k, else `SAQPrune`: `ScanHead` over every vector (the selection vector applied), then `ScanTail<W>` per chunk
  over the survivors against τ + margin_k (the last chunk at τ); tombstones drop at the merge. The query is int8 per
  chunk (`SAQQuery`). Filtered, by the cluster's passing fraction: below `SAQ_PRUNING_START_MAX_PASSING_FRACTION`
  (70%) the first clusters take `SAQPrune` too (τ = +∞ until the heap holds k), and below
  `SAQ_POSITIONS_HEAD_MAX_PASSING_FRACTION` (10%) its head is `ScanHeadPruning` over the passing positions. Without a
  PDX U8 head, `StartSAQEstimates`/`StartSAQEstimatesPruning` start the estimates at |r|² + |q_c|² (the head kernels do
  it themselves) and chunk 0 goes through `AccumulateTailEstimates`/`ScanTail` over every (or every passing) vector.
- **Pruning bound**: margin_k = ε_k·√(|q_c unseen|²·remaining variance). `IVFUtils::CalibrateSAQPruningBound` estimates
  ε at `saq_pruning_significance` (0.1), DADE-style (arXiv 2411.17229): the (1 − P_s) quantile of the unseen term's
  ratio over `SAQ_CALIBRATION_QUERIES` sampled rows, each against its `SAQ_CALIBRATION_NEIGHBORS` nearest neighbours by
  estimate in its `SAQ_CALIBRATION_CLUSTERS` nearest clusters. SAQ's own Chebyshev bound (§4.3) is not used: near
  neighbours' unseen term isn't centred at zero. P_s 0.1 against 0.01 costs 0.003–0.008 recall@10 at n_probe 128 on
  clip/jina/mxbai/openai and makes them 1.1–1.5× faster; it costs more where the head holds little of the variance
  (the near-isotropic test data: up to 0.035 probing every cluster). `CalibratePruningBound(row_ids, embeddings, n)` recalibrates on the
  caller's rows; `CalibratePruningBound()` on up to `SAQ_CALIBRATION_QUERIES` live rows sampled from the index
  (`SampleLiveRowIds`) and decoded, which calibrates as well as the original rows. The bound goes stale when appended
  rows differ from the ones it was calibrated on (rows stored in order); when to recalibrate is the caller's call.
  `IPDXIndex::CalibrateIndex()` (a no-op on other indexes) runs `CalibratePruningBound()` only if rows were appended
  since the last calibration (`IVF::dirty_calibration`, saved after ε, so a reloaded index keeps it), e.g. when the
  caller stops appending to the index.
- **Maintenance**: codes don't depend on the cluster (global mean, frozen model), so splits, steals, merges and
  `ReassignEmbeddings` move encoded rows as they are; decoded floats only drive the 2-means and the assignments.
  `GetEmbeddingsFromIndexByRowIds` returns decoded rows in T-space.
- **Tests** run SAQ at each suite's dimensions: mostly 128 (one segment) and 384 (the DP, free head). The PDX U8 head
  needs 640 dimensions, past the test data, so `test_saq` covers it on synthetic rows: it checks the plan in all three
  regimes and the cluster layout for both heads.

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

Header-only, no submodules; consumers link the `PDX` INTERFACE target (alias `PDX::PDX`), which carries `include/` and links `superkmeans::superkmeans`. SuperKMeans comes from FetchContent, pinned by `GIT_TAG` in `CMakeLists.txt` (`-DFETCHCONTENT_SOURCE_DIR_SUPERKMEANS=<checkout>` builds against a local one); its target brings Eigen, FFTW (optional), the executor and GEMM backends, compile definitions and the `-march`/MSVC/Emscripten flags. Benchmark binaries have **no** `.out` suffix.
```bash
cmake . -DPDX_COMPILE_BENCHMARKS=ON && make benchmarks
# Index building + search (index_type defaults to pdx_f32; nprobe 0/omitted sweeps a preset list)
# `index_type`: pdx_f32, pdx_u8, pdx_tree_f32, pdx_tree_u8, pdx_saq3, pdx_saq4 (every benchmark that takes an index type;
# BenchmarkPDXIVF and the filtered ones load what BenchmarkSerialization saved)
./benchmarks/BenchmarkEndToEnd <dataset_id> [index_type] [nprobe]
# Appends: build on n - n_append rows, append the rest in batches (1 = Append per row, 0 = rebuild reference)
./benchmarks/BenchmarkBulkAppend <dataset_id> <index_type> <batch_size> [n_append] [n_threads]
```
Add a benchmark with `pdx_add_benchmark(<Name> <source>)` in `benchmarks/CMakeLists.txt`; a test with `pdx_add_test(<name>.out <source>)` in `tests/CMakeLists.txt` (+ the `tests` custom target list).

Knobs: `-DPDX_MARCH` (default `native`, empty disables `-march`), `-DPDX_PORTABLE` (`-mavx2 -mfma` on x86_64 / plain `-O3` elsewhere, for wheels; also via the `PDX_PORTABLE` env var in `pip install .`), `-DPDX_SKIP_FFTW` (these three are forwarded to SuperKMeans' `SKMEANS_*`), `-DPDX_COMPILE_PYTHON` (bindings; defaults to ON only when PDX is the top-level project), and SuperKMeans' `-DSKMEANS_EXECUTOR` (`forkunion` default / `openmp` / `serial`) and `-DSKMEANS_GEMM` (`auto` default: Accelerate on macOS, Eigen elsewhere / `eigen` / `accelerate` / `blas`). See INSTALL.md.


## Code style

- **Naming**: `PascalCase` functions/classes/structs; `snake_case` variables/members;
  `UPPER_SNAKE_CASE` constants; everything lives in `namespace PDX` (the SuperKMeans dependency
  is `skmeans::`).
- Follow `.clang-format` / `.clang-tidy`.
- **Memory**: RAII, no raw `new`/`delete`. Buffers sized by n, n·d, a cluster's capacity·d, d·d or
  the worker count → `skmeans::AllocateBuffer`/`AllocateZeroedBuffer` on the bound allocator (see
  Memory allocator). Other buffers that don't need zero-init → `new T[]` in a `unique_ptr`,
  **not** `std::vector`/`resize()`.
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
  (`DistanceMetricRequiresNormalization`) and running L2SQ. The F32 computers also hold `FlipSign` and
  the LVQ centroid kernels: `HorizontalDotWithU8`/`U4` (float query · 8-bit or packed 4-bit codes),
  `HorizontalDotInt8WithU8`/`U4` (int8 query · codes: VPDPBUSD on AVX-512, and on AVX2 under `__AVXVNNI__`, else a
  widening `madd_epi16`; on NEON `vdotq_s32` under DOTPROD, LVQ8 on its codes XOR 0x80 plus 128·Σq (no i8mm: `-march=native`
can claim it on CPUs that lack it), else a widening `vmlal_s16`),
  `MinMax`, `QuantizeToU8`/`U4`, `DequantizeFromU4` and `QuantizeToInt8`. The SIMD backends hand their scalar
  remainders to `ScalarComputer` (`scalar_computer`). `DistanceComputer<L2SQ, SAQ>` holds SAQ's kernels on every backend:
  `ScanHead`, `ScanHeadPruning` (the head over given positions, gathered), `ScanTail<W>` (per-chunk table
  `GetScanTail(w)`), the no-pruning pair `ComputeHeadEstimates` /
  `AccumulateTailEstimates<W>`, the T2 `PackRecord` (`UnpackRecord` stays scalar), and the codec's `ComputeResidual`,
  `EncodeSegment` and `DecodeRecord` (float, 16 values per step; a segment whose length is not a multiple of 16 and a
  partial last chunk take the scalar kernels); the scans read and
  write up to `SAQ_SCAN_PADDING` lanes past the last vector.
  **All** SIMD lives in `include/pdx/distance_computers/` (`neon_computers.hpp`, `avx2_computers.hpp`,
  `avx512_computers.hpp`, `scalar_computers.hpp`) — don't scatter it elsewhere; keep all backends in
  sync when changing a kernel (on ARM the x86 ones aren't compiled or linted). Cover kernels in
  `tests/test_distance_computers.cpp`.
- **`PDX_VECTORIZE_LOOP`** (`common.hpp`) forces loop autovectorization (esp. FP reductions) — put it
  on its own line right above the `for`, never a raw `#pragma clang loop`. Other macros there: `PDX_RESTRICT`, `PDX_ALWAYS_INLINE`, `PDX_NO_INLINE`,
  `PDX_LIKELY`/`PDX_UNLIKELY`, `PDX_PREFETCH`, `PDX_POPCOUNT` (never a raw `__builtin_popcount`: MSVC
  lacks it), `PDX_ENSURE_POSITIVE`.
- **Parallelism: no OpenMP.** Every parallel loop runs on SuperKMeans' `ParallelExecutor` (aliased in
  `common.hpp`). `IPDXIndex` is a `skmeans::ExecutorHolder`: it binds `PDXIndexConfig::executor` or owns a
  default sized by `n_threads`, and `BuildIndex`/`Append`/`Delete` open a `skmeans::ParallelSection`. It
  hands the executor down: `IVFUtils(executor)` (normalize+rotate, k-means, cluster population),
  `ScalarQuantizer(d, &executor)`, `ADSamplingPruner::PreprocessEmbeddings(..., &executor)` (pruners are
  shared across indexes, so they never hold an index's executor) and `BatchComputer` calls (scratch of
  `ScratchSize(executor)` floats). `ComputeKMeans` with k=2 (the splits) runs `skmeans::TwoMeans` on the index's executor. A
  `ParallelFor` body must not call `ParallelFor`; FFTW plans only on the calling thread.
- **Memory allocator.** Large buffers come from SuperKMeans' `MemoryAllocator` (aliased in
  `common.hpp`), borrowed: it must outlive the index; unbound means the `VanillaAllocator`.
  `IPDXIndex` is a `skmeans::AllocatorHolder` bound to `PDXIndexConfig::allocator` (not serialized),
  and binds its `IVF` storage too (`IVF::SetAllocator` also binds its `Centroids`, `IVFTree::SetAllocator`
  L0; `Restore` and `BuildIndex`, which replace it, rebind it). It
  hands the allocator to its owned pruner, to `IVFUtils(executor, &allocator)` (which passes it to
  SuperKMeans' k-means) and to new clusters; each `Cluster` keeps it for `Grow`.
  `LoadPDXIndex`/`LoadPDXIndexFromStream` take it as a parameter, since loading allocates the clusters.
  Routed: cluster `data`/`indices`, the pruner's rotation (an `AllocatedBuffer` behind an `Eigen::Map`),
  the centroids (`Centroids`: the encoded centroids and their squared norms) and their decode scratch,
  `NormalizeAndRotate`, the population gathers, split/merge/reassign scratch
  (the split's A/B embeddings and ids, sized up front by `CountRowsToSteal`), `BulkAppend`'s row-id
  arrays, `ReadRotationMatrix`, `FlatIndex::GetEmbeddings`. On the heap: the k-means centroids
  (SuperKMeans' `Train` vector, until stored), `RowIdClusterMapping`, tombstones, `Flat` storage, per-query searcher buffers, the
  single-embedding transform in `Append`, d-sized and per-cluster (k-sized) arrays, the split's
  `SplitPartition` positions and k=2 assignments, `MmapFile` and Eigen's QR internals.
  `IPDXIndex::GetAllocatedSizeInBytes()` counts the routed buffers an index keeps (cluster data, centroids,
  an owned pruner's rotation); `GetSelfManagedMemorySizeInBytes()` is the rest of
  `GetInMemorySizeInBytes()` (what the index allocates itself, outside the allocator), for a caller whose
  allocator already charges the routed bytes.
- **Profiling**: `PDX_PROFILE_SCOPE("name")` (`profiler.hpp`); most benchmarks call
  `Profiler::Get().Print()` at the end (e.g. `BenchmarkEndToEnd`).
