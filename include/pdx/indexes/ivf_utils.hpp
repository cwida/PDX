#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pdx/clustering.hpp"
#include "pdx/common.hpp"
#include "pdx/distance_computers/base_computers.hpp"
#include "pdx/indexes/ivf_core.hpp"
#include "pdx/profiler.hpp"
#include "pdx/pruners/adsampling.hpp"
#include "pdx/quantizers/scalar.hpp"

namespace PDX {

struct PDXIndexConfig {
    uint32_t num_dimensions;
    DistanceMetric distance_metric = DistanceMetric::L2SQ;
    uint32_t seed = 42;
    uint32_t num_clusters = 0; // 0 = auto-compute from num_embeddings
    uint32_t num_meso_clusters = 0;
    bool normalize = false;
    float sampling_fraction = 0.0f; // 0 = auto (1.0 if small dataset, 0.3 otherwise)
    uint32_t kmeans_iters = 10;
    bool hierarchical_indexing = true;
    uint32_t n_threads = 0; // Threads of the default executor (0 = all cores); unused with executor
    bool is_data_transformed = false;
    size_t base_row_id = 0;
    uint32_t max_cluster_size = 0; // Clusters split at this size (0 = at 1.3x their initial size)
    bool build_cluster_sketches = false;
    bool expect_appends = true; // false: built clusters get one free slot (not serialized)
    CentroidQuantization centroid_quantization = CentroidQuantization::F32;
    ClusterStorageOrder cluster_storage_order = ClusterStorageOrder::NEAREST_CENTROID_CHAIN;
    uint32_t saq_bits_per_dimension = 3;
    float saq_pruning_significance = 0.1f;
    // Runs the parallel loops (not owned, not serialized). nullptr: the index creates a default
    // one.
    ParallelExecutor* executor = nullptr;
    // Allocates the large buffers (not owned, not serialized; must outlive the index). nullptr: the
    // VanillaAllocator.
    MemoryAllocator* allocator = nullptr;

    // Field by field with fixed widths, so the bytes do not depend on the struct's layout.
    void Save(std::ostream& out) const {
        WriteValue(out, num_dimensions);
        WriteValue(out, static_cast<uint8_t>(distance_metric));
        WriteValue(out, seed);
        WriteValue(out, num_clusters);
        WriteValue(out, num_meso_clusters);
        WriteValue(out, static_cast<uint8_t>(normalize));
        WriteValue(out, sampling_fraction);
        WriteValue(out, kmeans_iters);
        WriteValue(out, static_cast<uint8_t>(hierarchical_indexing));
        WriteValue(out, n_threads);
        WriteValue(out, static_cast<uint8_t>(is_data_transformed));
        WriteValue(out, static_cast<uint64_t>(base_row_id));
        WriteValue(out, max_cluster_size);
        WriteValue(out, static_cast<uint8_t>(build_cluster_sketches));
    }

    template <class Reader>
    void Load(Reader& reader) {
        num_dimensions = ReadValue<uint32_t>(reader);
        distance_metric = static_cast<DistanceMetric>(ReadValue<uint8_t>(reader));
        seed = ReadValue<uint32_t>(reader);
        num_clusters = ReadValue<uint32_t>(reader);
        num_meso_clusters = ReadValue<uint32_t>(reader);
        normalize = ReadValue<uint8_t>(reader) != 0;
        sampling_fraction = ReadValue<float>(reader);
        kmeans_iters = ReadValue<uint32_t>(reader);
        hierarchical_indexing = ReadValue<uint8_t>(reader) != 0;
        n_threads = ReadValue<uint32_t>(reader);
        is_data_transformed = ReadValue<uint8_t>(reader) != 0;
        base_row_id = static_cast<size_t>(ReadValue<uint64_t>(reader));
        max_cluster_size = ReadValue<uint32_t>(reader);
        build_cluster_sketches = ReadValue<uint8_t>(reader) != 0;
    }

    void Validate() const {
        if (num_dimensions == 0 || num_dimensions > PDX_MAX_DIMS) {
            throw std::invalid_argument(
                "num_dimensions must be between 1 and " + std::to_string(PDX_MAX_DIMS) + ", got " +
                std::to_string(num_dimensions)
            );
        }
        if (sampling_fraction < 0.0f || sampling_fraction > 1.0f) {
            throw std::invalid_argument(
                "sampling_fraction must be between 0.0 and 1.0, got " +
                std::to_string(sampling_fraction)
            );
        }
        if (num_meso_clusters > 0 && num_clusters > 0 && num_meso_clusters >= num_clusters) {
            throw std::invalid_argument(
                "num_meso_clusters (" + std::to_string(num_meso_clusters) +
                ") must be smaller than num_clusters (" + std::to_string(num_clusters) + ")"
            );
        }
        if (kmeans_iters == 0 || kmeans_iters >= 100) {
            throw std::invalid_argument(
                "kmeans_iters must be between 1 and 99, got " + std::to_string(kmeans_iters)
            );
        }
        if (max_cluster_size > 0 && max_cluster_size < MAX_CLUSTER_SIZE_LOWER_BOUND) {
            throw std::invalid_argument(
                "max_cluster_size must be 0 or at least " +
                std::to_string(MAX_CLUSTER_SIZE_LOWER_BOUND) + ", got " +
                std::to_string(max_cluster_size)
            );
        }
        if (saq_bits_per_dimension < SAQ_MIN_BITS_PER_DIMENSION ||
            saq_bits_per_dimension > SAQ_MAX_BITS_PER_DIMENSION) {
            throw std::invalid_argument(
                "saq_bits_per_dimension must be between " +
                std::to_string(SAQ_MIN_BITS_PER_DIMENSION) + " and " +
                std::to_string(SAQ_MAX_BITS_PER_DIMENSION) + ", got " +
                std::to_string(saq_bits_per_dimension)
            );
        }
        if (!(saq_pruning_significance > 0.0f && saq_pruning_significance < 1.0f)) {
            throw std::invalid_argument(
                "saq_pruning_significance must be between 0 and 1 (exclusive), got " +
                std::to_string(saq_pruning_significance)
            );
        }
    }

    void ValidateNumEmbeddings(size_t num_embeddings) const {
        if (num_clusters > 0 && num_clusters > num_embeddings) {
            throw std::invalid_argument(
                "num_clusters (" + std::to_string(num_clusters) + ") exceeds num_embeddings (" +
                std::to_string(num_embeddings) + ")"
            );
        }
    }
};

// The start of every SaveToStream stream, which LoadPDXIndexFromStream reads to construct the
// index.
inline void WriteStreamHeader(
    std::ostream& out,
    PDXIndexType type,
    CentroidQuantization centroid_quantization,
    const PDXIndexConfig& config
) {
    WriteValue(out, PDX_SERIALIZATION_VERSION);
    WriteValue(out, MakeIndexTypeByte(type, centroid_quantization));
    config.Save(out);
}

// The rotation of the file format (Save(path) / Restore(path)): its rows, its columns, then its
// values.
inline void WriteRotationMatrix(std::ostream& out, const ADSamplingPruner& pruner) {
    const auto& matrix = pruner.GetMatrix();
    WriteValue(out, static_cast<uint32_t>(matrix.rows()));
    WriteValue(out, static_cast<uint32_t>(matrix.cols()));
    out.write(
        reinterpret_cast<const char*>(matrix.data()),
        static_cast<std::streamsize>(sizeof(float) * matrix.rows() * matrix.cols())
    );
}

struct RotationMatrix {
    uint32_t rows;
    skmeans::AllocatedBuffer<float> values;
};

template <class Reader>
RotationMatrix ReadRotationMatrix(Reader& reader, MemoryAllocator& allocator) {
    const auto rows = ReadValue<uint32_t>(reader);
    const auto cols = ReadValue<uint32_t>(reader);
    const size_t num_values = static_cast<size_t>(rows) * cols;
    auto values = skmeans::AllocateBuffer<float>(allocator, num_values);
    reader.Read(values.get(), sizeof(float) * num_values);
    return {rows, std::move(values)};
}

// Dense in row id: PDX assigns row ids by position, so a sparse id space wastes memory here
struct RowIdClusterMapping {
    using entry_t = std::pair<uint32_t, uint32_t>;
    static_assert(sizeof(entry_t) == 2 * sizeof(uint32_t));
    static constexpr entry_t DELETED{DELETED_MARKER, DELETED_MARKER};

    void Set(size_t row_id, uint32_t cluster_id, uint32_t idx_in_cluster) {
        assert(row_id >= base_row_id);
        const size_t position = row_id - base_row_id;
        if (position >= entries.size()) {
            entries.resize(std::max(position + 1, entries.size() * 2), DELETED);
        }
        entries[position] = {cluster_id, idx_in_cluster};
    }

    void Delete(size_t row_id) {
        if (row_id >= base_row_id && row_id - base_row_id < entries.size()) {
            entries[row_id - base_row_id] = DELETED;
        }
    }

    [[nodiscard]] entry_t Get(size_t row_id) const {
        return row_id >= base_row_id && row_id - base_row_id < entries.size()
                   ? entries[row_id - base_row_id]
                   : DELETED;
    }

    template <class Clusters>
    void Rebuild(const Clusters& clusters, uint32_t num_clusters) {
        size_t size = 0;
        for (uint32_t c = 0; c < num_clusters; c++) {
            for (uint32_t p = 0; p < clusters[c].num_embeddings; p++) {
                size = std::max(size, clusters[c].indices[p] - base_row_id + 1);
            }
        }
        entries.assign(size, DELETED);
        for (uint32_t c = 0; c < num_clusters; c++) {
            for (uint32_t p = 0; p < clusters[c].num_embeddings; p++) {
                entries[clusters[c].indices[p] - base_row_id] = {c, p};
            }
        }
    }

    // Without the trailing deleted entries that Set's doubling leaves. base_row_id is in the
    // config.
    void Save(std::ostream& out) const {
        size_t num_entries = entries.size();
        while (num_entries > 0 && entries[num_entries - 1] == DELETED) {
            num_entries--;
        }
        WriteValue(out, static_cast<uint64_t>(num_entries));
        out.write(
            reinterpret_cast<const char*>(entries.data()),
            static_cast<std::streamsize>(num_entries * sizeof(entry_t))
        );
    }

    template <class Reader>
    void Load(Reader& reader) {
        entries.resize(static_cast<size_t>(ReadValue<uint64_t>(reader)));
        reader.Read(entries.data(), entries.size() * sizeof(entry_t));
    }

    [[nodiscard]] size_t SizeInBytes() const { return entries.size() * sizeof(entry_t); }

    size_t base_row_id = 0;
    std::vector<entry_t> entries;
};

// The index build steps that run in parallel, on the executor and allocator they are given
// (borrowed).
class IVFUtils : public skmeans::AllocatorHolder {
  public:
    explicit IVFUtils(ParallelExecutor& executor, MemoryAllocator* allocator = nullptr)
        : executor(executor) {
        SetAllocator(allocator);
    }

    skmeans::AllocatedBuffer<float> Normalize(
        const float* embeddings,
        size_t num_embeddings,
        uint32_t num_dimensions
    ) const {
        auto normalized =
            skmeans::AllocateBuffer<float>(GetAllocator(), num_embeddings * num_dimensions);
        Quantizer quantizer(num_dimensions);
        executor.ParallelFor(num_embeddings, [&](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; i++) {
                quantizer.NormalizeQuery(
                    embeddings + i * num_dimensions, normalized.get() + i * num_dimensions
                );
            }
        });
        return normalized;
    }

    skmeans::AllocatedBuffer<float> NormalizeAndRotate(
        const float* embeddings,
        size_t num_embeddings,
        uint32_t num_dimensions,
        bool normalize,
        const ADSamplingPruner& pruner
    ) const {
        PDX_PROFILE_SCOPE("Search/NormalizeAndRotate");
        const size_t total_floats = num_embeddings * num_dimensions;
        skmeans::AllocatedBuffer<float> normalized;
        const float* rotation_input = embeddings;
        if (normalize) {
            normalized = Normalize(embeddings, num_embeddings, num_dimensions);
            rotation_input = normalized.get();
        }
        auto preprocessed = skmeans::AllocateBuffer<float>(GetAllocator(), total_floats);
        pruner.PreprocessEmbeddings(rotation_input, preprocessed.get(), num_embeddings, &executor);
        return preprocessed;
    }

    // Compute centroids (clusters) and centroid-to-embedding assignments using SuperKMeans.
    [[nodiscard]] KMeansResult ComputeKMeans(
        const float* const embeddings,
        const uint64_t num_embeddings,
        const uint32_t num_dimensions,
        const uint32_t num_clusters,
        const PDX::DistanceMetric distance_metric,
        const uint32_t seed,
        const bool normalize = false,
        const float sampling_fraction = 0.0f,
        const uint32_t kmeans_iters = 8,
        const bool hierarchical_indexing = true,
        const bool use_superkmeans_pruning = true
    ) const {
        PDX_PROFILE_SCOPE("ComputeKMeans");
        assert(num_embeddings >= 1);
        assert(num_dimensions >= 1);
        assert(num_clusters >= 1);

        auto result = KMeansResult(num_clusters);

        if (num_clusters == 1) {
            result.centroids = std::vector<float>(embeddings, embeddings + num_dimensions);
            for (uint64_t vec_id = 0; vec_id < num_embeddings; vec_id++) {
                result.assignments[0].emplace_back(vec_id);
            }
            return result;
        }

        bool is_angular = normalize || distance_metric == PDX::DistanceMetric::COSINE ||
                          distance_metric == PDX::DistanceMetric::IP;

        float chosen_sampling_fraction = 0.3f;
        if (sampling_fraction > 0.0f) {
            chosen_sampling_fraction = sampling_fraction;
        } else if (num_embeddings < KMeansResult::MIN_EMBEDDINGS_TO_SAMPLE) {
            chosen_sampling_fraction = 1.0f;
        }

        bool use_hierarchical_indexing =
            hierarchical_indexing && num_embeddings >= KMeansResult::MIN_EMBEDDINGS_TO_SAMPLE;

        std::vector<uint32_t> assignments;
        if (num_clusters == 2) {
            result.centroids.resize(2 * static_cast<size_t>(num_dimensions));
            assignments.resize(num_embeddings);
            skmeans::TwoMeans(
                embeddings,
                num_embeddings,
                num_dimensions,
                kmeans_iters,
                TWO_MEANS_TOL,
                is_angular,
                seed,
                result.centroids.data(),
                assignments.data(),
                executor,
                GetAllocator()
            );
        } else if (use_hierarchical_indexing) {
            skmeans::HierarchicalSuperKMeansConfig config;
            config.sampling_fraction = 1.0f; // For now we are using all points
            config.angular = is_angular;
            config.data_already_rotated = true;
            config.use_blas_only = !use_superkmeans_pruning;
            config.suppress_warnings = true;
            config.iters_mesoclustering = 3;
            config.iters_fineclustering = 5;
            config.iters_refinement = 0;
            config.seed = seed;
            // config.verbose = true;
            config.executor = &executor;
            config.allocator = &GetAllocator();
            auto kmeans = skmeans::HierarchicalSuperKMeans(num_clusters, num_dimensions, config);
            result.centroids = kmeans.Train(embeddings, num_embeddings);
            if (num_clusters > skmeans::N_CLUSTERS_THRESHOLD_FOR_PRUNING) {
                assignments = kmeans.AssignTrainingPoints(
                    embeddings, result.centroids.data(), num_embeddings, num_clusters
                );
            } else {
                assignments = kmeans.Assign(
                    embeddings, result.centroids.data(), num_embeddings, num_clusters
                );
            }
        } else {
            skmeans::SuperKMeansConfig config;
            config.sampling_fraction = chosen_sampling_fraction;
            config.angular = is_angular;
            config.data_already_rotated = true;
            config.use_blas_only = !use_superkmeans_pruning;
            config.suppress_warnings = true;
            config.iters = kmeans_iters;
            config.seed = seed;
            // config.verbose = true;
            config.executor = &executor;
            config.allocator = &GetAllocator();
            auto kmeans = skmeans::SuperKMeans(num_clusters, num_dimensions, config);
            result.centroids = kmeans.Train(embeddings, num_embeddings);
            if (num_clusters > skmeans::N_CLUSTERS_THRESHOLD_FOR_PRUNING) {
                assignments = kmeans.AssignTrainingPoints(
                    embeddings, result.centroids.data(), num_embeddings, num_clusters
                );
            } else {
                assignments = kmeans.Assign(
                    embeddings, result.centroids.data(), num_embeddings, num_clusters
                );
            }
        }

        // Convert from vec_id -> centroid_idx into centroid_idx -> vec_id
        result.assignments.resize(num_clusters);
        for (uint64_t vec_id = 0; vec_id < num_embeddings; vec_id++) {
            result.assignments[assignments[vec_id]].emplace_back(vec_id);
        }

        return result;
    }

    // Splits the clusters of kmeans_result above max_cluster_size by k-means on their own
    // embeddings, into splits of about half of it, until they fit. A cluster k-means can't
    // separate (identical embeddings) stays whole.
    void SplitClustersAboveMaxClusterSize(
        KMeansResult& kmeans_result,
        const float* embeddings,
        const uint32_t num_dimensions,
        const uint32_t max_cluster_size,
        const PDX::DistanceMetric distance_metric,
        const uint32_t seed,
        const bool normalize,
        const bool use_superkmeans_pruning = true
    ) const {
        PDX_PROFILE_SCOPE("SplitClustersAboveMaxClusterSize");
        // The splits appended to kmeans_result are visited later in this loop
        for (size_t cluster_id = 0; cluster_id < kmeans_result.assignments.size(); cluster_id++) {
            bool splittable = true;
            while (splittable && kmeans_result.assignments[cluster_id].size() > max_cluster_size) {
                splittable = TrySplitClusterWithKMeans(
                    kmeans_result,
                    cluster_id,
                    embeddings,
                    num_dimensions,
                    max_cluster_size / 2, // target_size
                    distance_metric,
                    seed,
                    normalize,
                    use_superkmeans_pruning
                );
            }
        }
    }

    template <Quantization Q>
    std::unique_ptr<uint32_t[]> ComputeClusterStorageOrder(
        const IVF<Q>& ivf,
        const ClusterStorageOrder order,
        const uint32_t seed
    ) const {
        PDX_PROFILE_SCOPE("ComputeClusterStorageOrder");
        const uint32_t num_clusters = ivf.num_clusters;
        const uint32_t num_dimensions = ivf.num_dimensions;
        std::unique_ptr<uint32_t[]> storage_order(new uint32_t[num_clusters]);
        if (order == ClusterStorageOrder::ID) {
            for (uint32_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id) {
                storage_order[cluster_id] = cluster_id;
            }
            return storage_order;
        }
        // Empty clusters first: they hold no bytes, so they share the first cluster's offset
        size_t num_empty_clusters = 0;
        size_t last_unplaced_position = num_clusters;
        for (uint32_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id) {
            if (ivf.clusters[cluster_id].num_embeddings == 0) {
                storage_order[num_empty_clusters++] = cluster_id;
            } else {
                storage_order[--last_unplaced_position] = cluster_id;
            }
        }
        uint32_t* live_cluster_ids = storage_order.get() + num_empty_clusters;
        const size_t num_live_clusters = num_clusters - num_empty_clusters;
        if (num_live_clusters == 0) {
            return storage_order;
        }
        skmeans::AllocatedBuffer<float> decoded_centroids;
        const float* centroids = ivf.centroids.GetF32CentroidsPtr(decoded_centroids);
        if (num_live_clusters <= CLUSTER_STORAGE_ORDER_MAX_CHAIN_CLUSTERS) {
            ChainNearestCentroids(live_cluster_ids, num_live_clusters, centroids, num_dimensions);
            return storage_order;
        }
        // The path costs k^2 / 2 distances: above, one path per k-means group of the centroids,
        // the groups along a path over their own centroids
        auto live_centroids =
            skmeans::AllocateBuffer<float>(GetAllocator(), num_live_clusters * num_dimensions);
        std::unique_ptr<uint32_t[]> unordered_live_cluster_ids(new uint32_t[num_live_clusters]);
        for (size_t i = 0; i < num_live_clusters; ++i) {
            unordered_live_cluster_ids[i] = live_cluster_ids[i];
            std::memcpy(
                live_centroids.get() + i * num_dimensions,
                centroids + static_cast<size_t>(live_cluster_ids[i]) * num_dimensions,
                num_dimensions * sizeof(float)
            );
        }
        const auto num_groups =
            static_cast<uint32_t>(std::lround(std::sqrt(static_cast<double>(num_live_clusters))));
        const KMeansResult groups = ComputeKMeans(
            live_centroids.get(),
            num_live_clusters,
            num_dimensions,
            num_groups,
            DistanceMetric::L2SQ,
            seed
        );
        std::unique_ptr<uint32_t[]> group_order(new uint32_t[num_groups]);
        for (uint32_t group_id = 0; group_id < num_groups; ++group_id) {
            group_order[group_id] = group_id;
        }
        ChainNearestCentroids(
            group_order.get(), num_groups, groups.centroids.data(), num_dimensions
        );
        std::unique_ptr<size_t[]> group_offsets(new size_t[num_groups]);
        size_t group_offset = 0;
        for (uint32_t position = 0; position < num_groups; ++position) {
            group_offsets[position] = group_offset;
            group_offset += groups.assignments[group_order[position]].size();
        }
        executor.ParallelFor(num_groups, [&](size_t begin, size_t end, size_t) {
            for (size_t position = begin; position < end; ++position) {
                const auto& group_members = groups.assignments[group_order[position]];
                if (group_members.empty()) {
                    continue;
                }
                uint32_t* group_cluster_ids = live_cluster_ids + group_offsets[position];
                for (size_t i = 0; i < group_members.size(); ++i) {
                    group_cluster_ids[i] = unordered_live_cluster_ids[group_members[i]];
                }
                if (position > 0) {
                    const float* previous_group_centroid =
                        groups.centroids.data() +
                        static_cast<size_t>(group_order[position - 1]) * num_dimensions;
                    const size_t start = FindPositionOfNearestCentroidToCentroid(
                        group_cluster_ids,
                        0,
                        group_members.size(),
                        centroids,
                        num_dimensions,
                        previous_group_centroid
                    );
                    std::swap(group_cluster_ids[0], group_cluster_ids[start]);
                }
                ChainNearestCentroids(
                    group_cluster_ids, group_members.size(), centroids, num_dimensions
                );
            }
        });
        return storage_order;
    }

    template <Quantization Q>
    void PopulateIVFClusters(
        IVF<Q>& ivf,
        const KMeansResult& kmeans_result,
        const float* source_data,
        const size_t* row_ids,
        uint32_t num_dimensions,
        uint32_t num_clusters,
        float quantization_base,
        float quantization_scale,
        uint32_t max_cluster_size = 0,
        bool expect_appends = true
    ) const {
        PDX_PROFILE_SCOPE("PopulateIVFClusters");
        using storage_t = pdx_data_t<Q>;

        size_t largest_cluster_size = 0;
        for (size_t i = 0; i < num_clusters; i++) {
            largest_cluster_size =
                std::max(largest_cluster_size, kmeans_result.assignments[i].size());
        }

        // Pre-allocate all clusters sequentially
        const ClusterFormat<Q> cluster_format = ivf.GetClusterFormat();
        for (size_t cluster_idx = 0; cluster_idx < num_clusters; cluster_idx++) {
            const auto cluster_size =
                static_cast<uint32_t>(kmeans_result.assignments[cluster_idx].size());
            ivf.clusters.emplace_back(
                cluster_size,
                expect_appends ? Cluster<Q>::CapacityFor(cluster_size, max_cluster_size)
                               : cluster_size + 1,
                cluster_format,
                &GetAllocator()
            );
            ivf.clusters[cluster_idx].id = cluster_idx;
        }

        // Per-worker tmp buffers for gather + quantize
        const size_t n_workers = executor.NumWorkers();
        const size_t values_per_embedding = ivf.GetValuesPerEmbedding();
        std::vector<skmeans::AllocatedBuffer<storage_t>> tmp_buffers(n_workers);
        for (size_t w = 0; w < n_workers; w++) {
            tmp_buffers[w] = skmeans::AllocateBuffer<storage_t>(
                GetAllocator(), static_cast<uint64_t>(largest_cluster_size) * values_per_embedding
            );
        }
        const ClusterSketchQuantizer sketch_quantizer(cluster_format.sketch_num_dimensions);
        std::vector<skmeans::AllocatedBuffer<uint8_t>> sketch_buffers(n_workers);
        if (ivf.HasSketches()) {
            for (size_t w = 0; w < n_workers; w++) {
                sketch_buffers[w] = skmeans::AllocateBuffer<uint8_t>(
                    GetAllocator(), largest_cluster_size * sketch_quantizer.row_size_in_bytes
                );
            }
        }

        executor.ParallelFor(num_clusters, [&](size_t begin, size_t end, size_t worker) {
            auto* tmp = tmp_buffers[worker].get();
            uint8_t* sketch_rows = sketch_buffers[worker].get();
            const SAQQuantizer saq_quantizer(num_dimensions);
            std::unique_ptr<SAQEncodingScratch> saq_scratch;
            if constexpr (Q == SAQ) {
                saq_scratch = std::make_unique<SAQEncodingScratch>(num_dimensions);
            }
            for (size_t cluster_idx = begin; cluster_idx < end; cluster_idx++) {
                const auto cluster_size = kmeans_result.assignments[cluster_idx].size();
                auto& cluster = ivf.clusters[cluster_idx];

                for (size_t pos = 0; pos < cluster_size; pos++) {
                    const auto emb_idx = kmeans_result.assignments[cluster_idx][pos];
                    cluster.indices[pos] = row_ids[emb_idx];
                    if (sketch_rows != nullptr) {
                        sketch_quantizer.EncodeRow(
                            source_data + (emb_idx * num_dimensions),
                            ivf.sketch_mean.get(),
                            sketch_rows + pos * sketch_quantizer.row_size_in_bytes
                        );
                    }

                    if constexpr (Q == SAQ) {
                        saq_quantizer.EncodeEmbedding(
                            source_data + (emb_idx * num_dimensions),
                            ivf.saq_model,
                            tmp + (pos * values_per_embedding),
                            *saq_scratch
                        );
                    } else if constexpr (Q == U8) {
                        ScalarQuantizer<Q> quantizer(num_dimensions);
                        quantizer.QuantizeEmbedding(
                            source_data + (emb_idx * num_dimensions),
                            quantization_base,
                            quantization_scale,
                            tmp + (pos * num_dimensions)
                        );
                    } else {
                        std::memcpy(
                            tmp + (pos * num_dimensions),
                            source_data + (emb_idx * num_dimensions),
                            num_dimensions * sizeof(float)
                        );
                    }
                }
                cluster.StoreEmbeddings(tmp, cluster_size, sketch_rows);
            }
        });

        ivf.ComputeClusterOffsets();
    }

    // The pruning bound of an SAQ index. After 64-dim chunk k a vector is pruned when its running
    // estimate exceeds the k-th distance plus eps_k * |q_c unseen| * sqrt(unseen residual
    // variance), q_c being the query minus the mean. The running estimate already holds |q_c|^2 and
    // |r|^2: the final one is it minus 2 <q_c, y> over the unseen dimensions, which the margin must
    // cover. SAQ (Li et al., arXiv 2509.12086, Section 4.3) also prunes between segments, but
    // bounds that inner product with Chebyshev's inequality, as if it averaged zero. For the near
    // neighbours that pruning must keep it does not: they point towards the query, so that bound
    // drops them. As in DADE (Deng et al., arXiv 2411.17229), eps_k is estimated from the data at a
    // significance level instead: the (1 - significance) quantile of the ratio over rows of the
    // index as queries, each against its SAQ_CALIBRATION_NEIGHBORS nearest neighbours by estimate
    // in its SAQ_CALIBRATION_CLUSTERS nearest clusters (DADE samples pairs uniformly; these are the
    // pairs pruning must keep). pseudo_queries are transformed; their row ids (or null) keep each
    // one from being its own neighbour.
    void CalibrateSAQPruningBound(
        IVF<SAQ>& index,
        const float* pseudo_queries,
        const size_t* pseudo_query_row_ids,
        const size_t num_pseudo_queries,
        const float significance
    ) const {
        PDX_PROFILE_SCOPE("CalibrateSAQPruningBound");
        struct Candidate {
            float estimate;
            uint32_t cluster_id;
            uint32_t position;
        };
        const SAQModel& model = index.saq_model;
        const uint32_t d = index.num_dimensions;
        const uint32_t num_chunks = model.num_chunks;
        const float missing_ratio = std::numeric_limits<float>::quiet_NaN();
        // One slot per pseudo-query and neighbour in every chunk but the last
        const size_t ratios_per_chunk = num_pseudo_queries * SAQ_CALIBRATION_NEIGHBORS;
        auto ratios = skmeans::AllocateBuffer<float>(
            GetAllocator(), static_cast<size_t>(num_chunks - 1) * ratios_per_chunk
        );
        const ClusterFormat<SAQ> format = index.GetClusterFormat();
        const SAQQuantizer quantizer(d);
        const uint32_t num_nearest_clusters =
            std::min(SAQ_CALIBRATION_CLUSTERS, index.num_clusters);
        executor.ParallelFor(num_pseudo_queries, [&](size_t begin, size_t end, size_t) {
            auto estimates = skmeans::AllocateBuffer<float>(
                GetAllocator(), index.max_cluster_capacity + SAQ_SCAN_PADDING
            );
            auto candidates = skmeans::AllocateBuffer<Candidate>(
                GetAllocator(),
                static_cast<size_t>(num_nearest_clusters) * index.max_cluster_capacity
            );
            std::unique_ptr<float[]> cluster_distances(new float[index.num_clusters]);
            std::unique_ptr<uint32_t[]> cluster_order(new uint32_t[index.num_clusters]);
            std::unique_ptr<float[]> query_residual(new float[d]);
            std::unique_ptr<uint8_t[]> encoded(new uint8_t[index.GetValuesPerEmbedding()]);
            std::unique_ptr<float[]> decoded(new float[d]);
            std::unique_ptr<float[]> chunk_dots(new float[num_chunks]);
            for (size_t query_idx = begin; query_idx < end; ++query_idx) {
                const float* pseudo_query = pseudo_queries + query_idx * d;
                const SAQQuery query(pseudo_query, model);
                for (uint32_t j = 0; j < d; ++j) {
                    query_residual[j] = pseudo_query[j] - model.rotated_mean[j];
                }
                // TODO(@lkuffo, low): Rank every pseudo-query at once with one blocked GEMM top-k
                // (BatchComputer::FindKNearestNeighbors over the decoded centroids): this streams
                // all K centroids per pseudo-query, 2 s F32 / 0.6 s LVQ8 at 100K clusters
                index.centroids.ComputeDistancesToCentroids(pseudo_query, cluster_distances.get());
                std::iota(cluster_order.get(), cluster_order.get() + index.num_clusters, 0u);
                std::partial_sort(
                    cluster_order.get(),
                    cluster_order.get() + num_nearest_clusters,
                    cluster_order.get() + index.num_clusters,
                    [&](uint32_t a, uint32_t b) {
                        return cluster_distances[a] < cluster_distances[b];
                    }
                );
                size_t num_candidates = 0;
                for (uint32_t nearest_idx = 0; nearest_idx < num_nearest_clusters; ++nearest_idx) {
                    const uint32_t cluster_id = cluster_order[nearest_idx];
                    const auto& cluster = index.clusters[cluster_id];
                    if (cluster.used_capacity == 0) {
                        continue;
                    }
                    ComputeSAQEstimates(
                        cluster.data.get(),
                        cluster.used_capacity,
                        cluster.max_capacity,
                        model,
                        query,
                        estimates.get()
                    );
                    for (uint32_t position = 0; position < cluster.used_capacity; ++position) {
                        const bool is_deleted =
                            !cluster.tombstones.empty() && cluster.tombstones.count(position) != 0;
                        const bool is_pseudo_query =
                            pseudo_query_row_ids != nullptr &&
                            cluster.indices[position] == pseudo_query_row_ids[query_idx];
                        if (!is_deleted && !is_pseudo_query) {
                            candidates[num_candidates++] = {
                                estimates[position], cluster_id, position
                            };
                        }
                    }
                }
                const size_t num_neighbors = std::min(SAQ_CALIBRATION_NEIGHBORS, num_candidates);
                std::partial_sort(
                    candidates.get(),
                    candidates.get() + num_neighbors,
                    candidates.get() + num_candidates,
                    [](const Candidate& a, const Candidate& b) { return a.estimate < b.estimate; }
                );
                for (size_t neighbor_idx = 0; neighbor_idx < SAQ_CALIBRATION_NEIGHBORS;
                     ++neighbor_idx) {
                    float* neighbor_ratios =
                        ratios.get() + query_idx * SAQ_CALIBRATION_NEIGHBORS + neighbor_idx;
                    if (neighbor_idx >= num_neighbors) {
                        for (uint32_t chunk_idx = 0; chunk_idx + 1 < num_chunks; ++chunk_idx) {
                            neighbor_ratios[chunk_idx * ratios_per_chunk] = missing_ratio;
                        }
                        continue;
                    }
                    const Candidate& neighbor = candidates[neighbor_idx];
                    const auto& cluster = index.clusters[neighbor.cluster_id];
                    Cluster<SAQ>::ReadEmbeddingFromPDXBuffer(
                        cluster.data.get(),
                        cluster.max_capacity,
                        format,
                        neighbor.position,
                        encoded.get()
                    );
                    quantizer.DecodeEmbedding(encoded.get(), model, decoded.get());
                    for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
                        const uint32_t first_dimension = chunk_idx * SAQ_CHUNK_SIZE;
                        const uint32_t last_dimension =
                            std::min(first_dimension + SAQ_CHUNK_SIZE, d);
                        float dot = 0.0f;
                        for (uint32_t j = first_dimension; j < last_dimension; ++j) {
                            dot += query_residual[j] * (decoded[j] - model.rotated_mean[j]);
                        }
                        chunk_dots[chunk_idx] = dot;
                    }
                    // The unseen term and its spread after each chunk, as the search's margins
                    float unseen_term = 0.0f;
                    float unseen_squared_norm = 0.0f;
                    for (int chunk_idx = static_cast<int>(num_chunks) - 2; chunk_idx >= 0;
                         --chunk_idx) {
                        unseen_term += 2.0f * chunk_dots[chunk_idx + 1];
                        unseen_squared_norm += query.GetChunkSquaredNormsPtr()[chunk_idx + 1];
                        const float spread = std::sqrt(
                            unseen_squared_norm * model.chunk_remaining_variances[chunk_idx]
                        );
                        neighbor_ratios[static_cast<size_t>(chunk_idx) * ratios_per_chunk] =
                            spread > 0.0f ? unseen_term / spread : missing_ratio;
                    }
                }
            }
        });

        // The quantile with linear interpolation between the sorted ratios
        for (uint32_t chunk_idx = 0; chunk_idx + 1 < num_chunks; ++chunk_idx) {
            float* chunk_ratios = ratios.get() + static_cast<size_t>(chunk_idx) * ratios_per_chunk;
            float* chunk_ratios_end =
                std::remove_if(chunk_ratios, chunk_ratios + ratios_per_chunk, [](float r) {
                    return std::isnan(r);
                });
            const auto num_ratios = static_cast<size_t>(chunk_ratios_end - chunk_ratios);
            if (num_ratios == 0) {
                index.saq_chunk_epsilons[chunk_idx] = 0.0f;
                continue;
            }
            std::sort(chunk_ratios, chunk_ratios_end);
            const double rank = (1.0 - significance) * static_cast<double>(num_ratios - 1);
            const auto lower = static_cast<size_t>(rank);
            const size_t upper = std::min(lower + 1, num_ratios - 1);
            index.saq_chunk_epsilons[chunk_idx] = static_cast<float>(
                chunk_ratios[lower] +
                (chunk_ratios[upper] - chunk_ratios[lower]) * (rank - static_cast<double>(lower))
            );
        }
        // The last chunk prunes at the k-th distance itself
        index.saq_chunk_epsilons[num_chunks - 1] = 0.0f;
    }

    void AssignToNearestCentroids(
        const float* embeddings,
        const size_t num_embeddings,
        const float* centroids,
        const float* centroid_norms,
        const size_t num_centroids,
        const uint32_t num_dimensions,
        uint32_t* out_assignments
    ) const {
        using batch_computer =
            skmeans::BatchComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::f32>;
        // TODO(@lkuffo, mid): An embedding with NaNs never beats a distance, so its assignment is
        // never written. We assume no NaNs.
        auto embedding_norms = skmeans::AllocateBuffer<float>(GetAllocator(), num_embeddings);
        Eigen::Map<Eigen::VectorXf>(embedding_norms.get(), num_embeddings).noalias() =
            Eigen::Map<const eigen_matrix_t>(embeddings, num_embeddings, num_dimensions)
                .rowwise()
                .squaredNorm();
        auto distances = skmeans::AllocateBuffer<float>(GetAllocator(), num_embeddings);
        const size_t n_workers = executor.NumWorkers();
        // Fewer 256-row blocks than workers: each worker takes a slice of the centroids instead
        if (n_workers > 1 && num_embeddings < skmeans::MINI_BATCH_SIZE * n_workers) {
            const size_t n_slices = std::min(n_workers, num_centroids);
            skmeans::SerialExecutor serial_executor;
            const size_t slice_scratch_size = batch_computer::ScratchSize(serial_executor);
            auto slice_assignments =
                skmeans::AllocateBuffer<uint32_t>(GetAllocator(), n_slices * num_embeddings);
            auto slice_distances =
                skmeans::AllocateBuffer<float>(GetAllocator(), n_slices * num_embeddings);
            auto slice_scratch =
                skmeans::AllocateBuffer<float>(GetAllocator(), n_slices * slice_scratch_size);
            executor.ParallelFor(n_slices, [&](size_t begin, size_t end, size_t) {
                skmeans::SerialExecutor slice_executor;
                for (size_t s = begin; s < end; s++) {
                    const size_t first = num_centroids * s / n_slices;
                    const size_t last = num_centroids * (s + 1) / n_slices;
                    batch_computer::FindNearestNeighbor(
                        slice_executor,
                        embeddings,
                        centroids + first * num_dimensions,
                        num_embeddings,
                        last - first,
                        num_dimensions,
                        embedding_norms.get(),
                        centroid_norms + first,
                        slice_assignments.get() + s * num_embeddings,
                        slice_distances.get() + s * num_embeddings,
                        slice_scratch.get() + s * slice_scratch_size
                    );
                }
            });
            for (size_t i = 0; i < num_embeddings; i++) {
                size_t best_slice = 0;
                for (size_t s = 1; s < n_slices; s++) {
                    if (slice_distances[s * num_embeddings + i] <
                        slice_distances[best_slice * num_embeddings + i]) {
                        best_slice = s;
                    }
                }
                out_assignments[i] = static_cast<uint32_t>(num_centroids * best_slice / n_slices) +
                                     slice_assignments[best_slice * num_embeddings + i];
            }
            return;
        }
        auto scratch =
            skmeans::AllocateBuffer<float>(GetAllocator(), batch_computer::ScratchSize(executor));
        batch_computer::FindNearestNeighbor(
            executor,
            embeddings,
            centroids,
            num_embeddings,
            num_centroids,
            num_dimensions,
            embedding_norms.get(),
            centroid_norms,
            out_assignments,
            distances.get(),
            scratch.get()
        );
    }

    static void GroupRowsByCluster(
        const uint32_t* assignments,
        const size_t num_rows,
        const uint32_t num_clusters,
        std::vector<uint32_t>& cluster_starts,
        uint32_t* rows_by_cluster
    ) {
        cluster_starts.assign(static_cast<size_t>(num_clusters) + 1, 0);
        for (size_t i = 0; i < num_rows; i++) {
            cluster_starts[assignments[i] + 1]++;
        }
        std::partial_sum(cluster_starts.begin(), cluster_starts.end(), cluster_starts.begin());
        std::vector<uint32_t> next_slot(cluster_starts.begin(), cluster_starts.end() - 1);
        for (size_t i = 0; i < num_rows; i++) {
            rows_by_cluster[next_slot[assignments[i]]++] = static_cast<uint32_t>(i);
        }
    }

    struct ClusterAppendRange {
        uint32_t cluster_id;
        uint32_t first_row; // Position in rows_by_cluster
        uint32_t num_rows;
    };

    template <Quantization Q>
    void AppendRowsToClusters(
        IVF<Q>& ivf,
        const float* embeddings,
        const uint32_t* row_ids,
        const uint32_t* rows_by_cluster,
        const std::vector<ClusterAppendRange>& ranges,
        uint32_t* out_positions
    ) const {
        using storage_t = pdx_data_t<Q>;
        const size_t d = ivf.num_dimensions;
        size_t largest_range = 0;
        for (const auto& range : ranges) {
            largest_range = std::max<size_t>(largest_range, range.num_rows);
        }
        const size_t n_workers = executor.NumWorkers();
        const size_t values_per_embedding = ivf.GetValuesPerEmbedding();
        auto rows_buffer = skmeans::AllocateBuffer<storage_t>(
            GetAllocator(), n_workers * largest_range * values_per_embedding
        );
        auto ids_buffer =
            skmeans::AllocateBuffer<uint32_t>(GetAllocator(), n_workers * largest_range);
        auto positions_buffer =
            skmeans::AllocateBuffer<uint32_t>(GetAllocator(), n_workers * largest_range);
        const ClusterSketchQuantizer sketch_quantizer(ivf.HasSketches() ? ivf.num_dimensions : 0);
        skmeans::AllocatedBuffer<uint8_t> sketches_buffer;
        if (ivf.HasSketches()) {
            sketches_buffer = skmeans::AllocateBuffer<uint8_t>(
                GetAllocator(), n_workers * largest_range * sketch_quantizer.row_size_in_bytes
            );
        }
        executor.ParallelFor(ranges.size(), [&](size_t begin, size_t end, size_t worker) {
            storage_t* rows = rows_buffer.get() + worker * largest_range * values_per_embedding;
            uint32_t* ids = ids_buffer.get() + worker * largest_range;
            uint32_t* positions = positions_buffer.get() + worker * largest_range;
            uint8_t* sketch_rows =
                ivf.HasSketches()
                    ? sketches_buffer.get() + worker * largest_range * sketch_quantizer.row_size_in_bytes
                    : nullptr;
            ScalarQuantizer<Q> quantizer(d);
            const SAQQuantizer saq_quantizer(d);
            std::unique_ptr<SAQEncodingScratch> saq_scratch;
            if constexpr (Q == SAQ) {
                saq_scratch = std::make_unique<SAQEncodingScratch>(d);
            }
            for (size_t r = begin; r < end; r++) {
                const ClusterAppendRange& range = ranges[r];
                for (uint32_t j = 0; j < range.num_rows; j++) {
                    const uint32_t row = rows_by_cluster[range.first_row + j];
                    ids[j] = row_ids[row];
                    if (sketch_rows != nullptr) {
                        sketch_quantizer.EncodeRow(
                            embeddings + row * d,
                            ivf.sketch_mean.get(),
                            sketch_rows + j * sketch_quantizer.row_size_in_bytes
                        );
                    }
                    if constexpr (Q == SAQ) {
                        saq_quantizer.EncodeEmbedding(
                            embeddings + row * d,
                            ivf.saq_model,
                            rows + j * values_per_embedding,
                            *saq_scratch
                        );
                    } else if constexpr (Q == U8) {
                        quantizer.QuantizeEmbedding(
                            embeddings + row * d,
                            ivf.quantization_base,
                            ivf.quantization_scale,
                            rows + j * values_per_embedding
                        );
                    } else {
                        std::memcpy(
                            rows + j * values_per_embedding, embeddings + row * d, d * sizeof(float)
                        );
                    }
                }
                ivf.clusters[range.cluster_id].AppendEmbeddings(
                    rows, ids, range.num_rows, positions, sketch_rows
                );
                for (uint32_t j = 0; j < range.num_rows; j++) {
                    out_positions[rows_by_cluster[range.first_row + j]] = positions[j];
                }
            }
        });
    }

  private:
    // Reorders cluster_ids into a greedy nearest-neighbour path over their centroids, from the
    // first one
    static void ChainNearestCentroids(
        uint32_t* cluster_ids,
        const size_t num_cluster_ids,
        const float* centroids,
        const uint32_t num_dimensions
    ) {
        for (size_t position = 1; position < num_cluster_ids; ++position) {
            const float* previous_centroid =
                centroids + static_cast<size_t>(cluster_ids[position - 1]) * num_dimensions;
            const size_t nearest_position = FindPositionOfNearestCentroidToCentroid(
                cluster_ids, position, num_cluster_ids, centroids, num_dimensions, previous_centroid
            );
            std::swap(cluster_ids[position], cluster_ids[nearest_position]);
        }
    }

    static size_t FindPositionOfNearestCentroidToCentroid(
        const uint32_t* cluster_ids,
        const size_t begin,
        const size_t end,
        const float* centroids,
        const uint32_t num_dimensions,
        const float* centroid
    ) {
        size_t nearest_position = begin;
        float nearest_distance = std::numeric_limits<float>::max();
        for (size_t position = begin; position < end; ++position) {
            const float distance = DistanceComputer<DistanceMetric::L2SQ, F32>::Horizontal(
                centroid,
                centroids + static_cast<size_t>(cluster_ids[position]) * num_dimensions,
                num_dimensions
            );
            if (distance < nearest_distance) {
                nearest_distance = distance;
                nearest_position = position;
            }
        }
        return nearest_position;
    }

    // Splits one cluster of kmeans_result by k-means on its embeddings into non-empty splits of
    // about target_size. False if k-means leaves it whole.
    bool TrySplitClusterWithKMeans(
        KMeansResult& kmeans_result,
        const size_t cluster_id,
        const float* embeddings,
        const uint32_t num_dimensions,
        const uint32_t target_size,
        const PDX::DistanceMetric distance_metric,
        const uint32_t seed,
        const bool normalize,
        const bool use_superkmeans_pruning
    ) const {
        const size_t d = num_dimensions;
        const auto& ids = kmeans_result.assignments[cluster_id];
        const size_t num_embeddings = ids.size();
        auto cluster_embeddings =
            skmeans::AllocateBuffer<float>(GetAllocator(), num_embeddings * d);
        for (size_t i = 0; i < num_embeddings; i++) {
            std::memcpy(
                cluster_embeddings.get() + i * d, embeddings + ids[i] * d, d * sizeof(float)
            );
        }
        const auto num_splits =
            static_cast<uint32_t>((num_embeddings + target_size - 1) / target_size);
        KMeansResult splits = ComputeKMeans(
            cluster_embeddings.get(),
            num_embeddings,
            num_dimensions,
            num_splits,
            distance_metric,
            seed,
            normalize,
            1.0f,
            SPLIT_KMEANS_ITERS,
            false,
            use_superkmeans_pruning
        );

        // From positions in the cluster to positions in embeddings
        std::vector<uint32_t> non_empty_splits;
        for (uint32_t split = 0; split < num_splits; split++) {
            if (splits.assignments[split].empty()) {
                continue;
            }
            non_empty_splits.push_back(split);
            for (auto& position : splits.assignments[split]) {
                position = ids[position];
            }
        }
        if (non_empty_splits.size() < 2) {
            return false;
        }
        for (size_t i = 0; i < non_empty_splits.size(); i++) {
            const uint32_t split = non_empty_splits[i];
            const float* centroid = splits.centroids.data() + static_cast<size_t>(split) * d;
            // The first non-empty split replaces the cluster being split in its slot; the others
            // are appended
            if (i == 0) {
                kmeans_result.assignments[cluster_id] = std::move(splits.assignments[split]);
                std::memcpy(
                    kmeans_result.centroids.data() + cluster_id * d, centroid, d * sizeof(float)
                );
            } else {
                kmeans_result.assignments.push_back(std::move(splits.assignments[split]));
                kmeans_result.centroids.insert(
                    kmeans_result.centroids.end(), centroid, centroid + d
                );
            }
        }
        return true;
    }

    ParallelExecutor& executor;
};

// ******************************************
// Maintenance helpers (SPFresh-like Append/Delete), shared by PDXIndex and PDXTreeIndex.
// ******************************************

// Dequantize raw (Q-type) embeddings to float. For F32 this is a memcpy.
template <Quantization Q>
inline skmeans::AllocatedBuffer<float> DequantizeClusterEmbeddings(
    const IVF<Q>& index,
    ScalarQuantizer<Q>& quantizer,
    const pdx_data_t<Q>* raw_embeddings,
    uint32_t n_emb,
    MemoryAllocator& allocator
) {
    PDX_PROFILE_SCOPE("Dequantize");
    const size_t d = index.num_dimensions;
    auto result = skmeans::AllocateBuffer<float>(allocator, static_cast<size_t>(n_emb) * d);
    if constexpr (Q == SAQ) {
        const SAQQuantizer saq_quantizer(d);
        const size_t values_per_embedding = index.GetValuesPerEmbedding();
        for (size_t i = 0; i < n_emb; i++) {
            saq_quantizer.DecodeEmbedding(
                raw_embeddings + i * values_per_embedding, index.saq_model, result.get() + i * d
            );
        }
    } else if constexpr (Q == U8) {
        for (size_t i = 0; i < n_emb; i++) {
            quantizer.DequantizeEmbedding(
                raw_embeddings + i * d,
                index.quantization_base,
                index.quantization_scale,
                result.get() + i * d
            );
        }
    } else if (n_emb > 0) {
        std::memcpy(result.get(), raw_embeddings, static_cast<size_t>(n_emb) * d * sizeof(float));
    }
    return result;
}

// The body of GetEmbeddingsFromIndexByRowIds shared by PDXIndex and PDXTreeIndex: each row read
// from its cluster's slot (u8 dequantized), in the order of row_ids. The rows are read cluster by
// cluster, so that a cluster without data is acquired from cluster_source once.
template <Quantization Q>
inline void GetEmbeddingsFromIndexByRowIdsImpl(
    const IVF<Q>& index,
    [[maybe_unused]] ScalarQuantizer<Q>& quantizer,
    const RowIdClusterMapping& row_id_cluster_mapping,
    const std::vector<size_t>& row_ids,
    float* out,
    IClusterSource* cluster_source
) {
    using data_t = pdx_data_t<Q>;
    const size_t d = index.num_dimensions;
    std::vector<RowIdClusterMapping::entry_t> locations(row_ids.size());
    for (size_t i = 0; i < row_ids.size(); i++) {
        locations[i] = row_id_cluster_mapping.Get(row_ids[i]);
        if (locations[i].first == DELETED_MARKER) {
            throw std::invalid_argument(
                "GetEmbeddingsFromIndexByRowIds: a row id is not in the index"
            );
        }
    }
    std::vector<size_t> rows_by_cluster(row_ids.size());
    std::iota(rows_by_cluster.begin(), rows_by_cluster.end(), 0);
    std::sort(rows_by_cluster.begin(), rows_by_cluster.end(), [&](size_t a, size_t b) {
        return locations[a].first < locations[b].first;
    });
    std::unique_ptr<data_t[]> embedding(new data_t[index.GetValuesPerEmbedding()]);
    const ClusterFormat<Q> cluster_format = index.GetClusterFormat();
    const SAQQuantizer saq_quantizer(d);
    size_t next = 0;
    while (next < rows_by_cluster.size()) {
        const uint32_t cluster_id = locations[rows_by_cluster[next]].first;
        const auto& cluster = index.clusters[cluster_id];
        const bool from_cluster_source = !cluster.data;
        const data_t* data = cluster.data.get();
        size_t stride = cluster.max_capacity;
        if (from_cluster_source) {
            const char* cluster_bytes = cluster_source->Acquire(cluster_id);
            data = reinterpret_cast<const data_t*>(
                cluster_bytes + sizeof(uint32_t) * cluster.used_capacity
            );
            stride = cluster.used_capacity;
        }
        for (;
             next < rows_by_cluster.size() && locations[rows_by_cluster[next]].first == cluster_id;
             next++) {
            const size_t i = rows_by_cluster[next];
            Cluster<Q>::ReadEmbeddingFromPDXBuffer(
                data, stride, cluster_format, locations[i].second, embedding.get()
            );
            if constexpr (Q == SAQ) {
                saq_quantizer.DecodeEmbedding(embedding.get(), index.saq_model, out + i * d);
            } else if constexpr (Q == U8) {
                quantizer.DequantizeEmbedding(
                    embedding.get(), index.quantization_base, index.quantization_scale, out + i * d
                );
            } else {
                std::copy(embedding.get(), embedding.get() + d, out + i * d);
            }
        }
        if (from_cluster_source) {
            cluster_source->Release(cluster_id);
        }
    }
}

// Quantize (if U8) and append a float embedding to a cluster. Returns its index in the cluster.
template <Quantization Q>
inline uint32_t QuantizeAndAppend(
    const IVF<Q>& index,
    ScalarQuantizer<Q>& quantizer,
    Cluster<Q>& cluster,
    uint32_t row_id,
    const float* embedding
) {
    std::unique_ptr<uint8_t[]> sketch_row;
    if (index.HasSketches()) {
        const ClusterSketchQuantizer sketch_quantizer(index.num_dimensions);
        sketch_row.reset(new uint8_t[sketch_quantizer.row_size_in_bytes]);
        sketch_quantizer.EncodeRow(embedding, index.sketch_mean.get(), sketch_row.get());
    }
    if constexpr (Q == SAQ) {
        std::unique_ptr<uint8_t[]> encoded(new uint8_t[index.GetValuesPerEmbedding()]);
        SAQEncodingScratch scratch(index.num_dimensions);
        SAQQuantizer(index.num_dimensions)
            .EncodeEmbedding(embedding, index.saq_model, encoded.get(), scratch);
        return cluster.AppendEmbedding(row_id, encoded.get(), sketch_row.get());
    } else if constexpr (Q == U8) {
        std::unique_ptr<pdx_data_t<Q>[]> quantized(new pdx_data_t<Q>[index.num_dimensions]);
        quantizer.QuantizeEmbedding(
            embedding, index.quantization_base, index.quantization_scale, quantized.get()
        );
        return cluster.AppendEmbedding(row_id, quantized.get(), sketch_row.get());
    } else {
        return cluster.AppendEmbedding(row_id, embedding, sketch_row.get());
    }
}

// Gather the raw embeddings and row ids at the given positions of a cluster into embs_out and
// ids_out (room for group_idx.size() rows), accumulating their (float) sum into centroid_sum.
template <Quantization Q>
inline void GatherGroupEmbeddings(
    const IVF<Q>& index,
    const Cluster<Q>& cluster,
    const std::vector<uint32_t>& group_idx,
    const pdx_data_t<Q>* raw_embeddings,
    const float* float_embeddings,
    pdx_data_t<Q>* embs_out,
    uint32_t* ids_out,
    float* centroid_sum,
    const uint8_t* sketch_rows = nullptr,
    uint8_t* sketches_out = nullptr
) {
    const size_t d = index.num_dimensions;
    const size_t values_per_embedding = index.GetValuesPerEmbedding();
    const size_t sketch_row_size = ClusterSketchQuantizer(index.num_dimensions).row_size_in_bytes;
    for (size_t i = 0; i < group_idx.size(); i++) {
        const uint32_t idx = group_idx[i];
        std::memcpy(
            embs_out + i * values_per_embedding,
            raw_embeddings + static_cast<size_t>(idx) * values_per_embedding,
            values_per_embedding * sizeof(pdx_data_t<Q>)
        );
        if (sketch_rows != nullptr) {
            std::memcpy(
                sketches_out + i * sketch_row_size,
                sketch_rows + static_cast<size_t>(idx) * sketch_row_size,
                sketch_row_size
            );
        }
        ids_out[i] = cluster.indices[idx];
        const float* emb_f = float_embeddings + static_cast<size_t>(idx) * d;
        for (size_t j = 0; j < d; j++) {
            centroid_sum[j] += emb_f[j];
        }
    }
}

// Mean centroid from an accumulated sum; falls back to `fallback` when count == 0.
// Re-normalized when the index stores normalized vectors.
inline void ComputeCentroidMean(
    uint32_t num_dimensions,
    bool normalize,
    const float* centroid_sum,
    size_t count,
    const float* fallback,
    float* output
) {
    if (count == 0) {
        std::memcpy(output, fallback, num_dimensions * sizeof(float));
    } else {
        float inv = 1.0f / static_cast<float>(count);
        PDX_VECTORIZE_LOOP
        for (size_t j = 0; j < num_dimensions; j++) {
            output[j] = centroid_sum[j] * inv;
        }
    }
    if (normalize) {
        Quantizer q(num_dimensions);
        q.NormalizeQuery(output, output);
    }
}

struct SplitPartition {
    std::unique_ptr<float[]> centroid_a;
    std::unique_ptr<float[]> centroid_b;
    std::vector<uint32_t> group_a_idx;
    std::vector<uint32_t> group_b_idx;
    std::vector<uint32_t> group_rest_idx; // closer to a neighboring cluster than to A or B
};

// 2-means over the (float) embeddings of a cluster that is about to split. Every position lands
// in group A, group B, or "rest" when one of the neighboring clusters' centroids is closer than
// both A and B (the caller reassigns those).
template <Quantization Q>
inline SplitPartition PartitionClusterForSplit(
    const IVF<Q>& index,
    const float* cluster_embeddings,
    uint32_t num_embeddings,
    const float* centroid_to_split,
    const std::vector<uint32_t>& neighboring_clusters_ids,
    DistanceMetric distance_metric,
    uint32_t seed,
    ParallelExecutor& executor,
    MemoryAllocator& allocator
) {
    using distance_computer_f32_t = DistanceComputer<DistanceMetric::L2SQ, F32>;
    const size_t d = index.num_dimensions;
    SplitPartition partition;
    partition.centroid_a.reset(new float[d]);
    partition.centroid_b.reset(new float[d]);
    {
        PDX_PROFILE_SCOPE("Split/KMeans");
        KMeansResult split_result = IVFUtils(executor, &allocator)
                                        .ComputeKMeans(
                                            cluster_embeddings,
                                            num_embeddings,
                                            d,
                                            2,
                                            distance_metric,
                                            seed,
                                            index.is_normalized,
                                            1.0f,
                                            SPLIT_KMEANS_ITERS,
                                            false
                                        );
        std::memcpy(partition.centroid_a.get(), split_result.centroids.data(), d * sizeof(float));
        std::memcpy(
            partition.centroid_b.get(), split_result.centroids.data() + d, d * sizeof(float)
        );
        // A and B are compared with decoded centroids below, so they carry the same error
        index.centroids.RoundTripThroughEncoding(partition.centroid_a.get());
        index.centroids.RoundTripThroughEncoding(partition.centroid_b.get());
        partition.group_a_idx.reserve(split_result.assignments[0].size());
        partition.group_b_idx.reserve(split_result.assignments[1].size());
    }

    // Assign each embedding to A, B, or rest (closer elsewhere)
    {
        PDX_PROFILE_SCOPE("Split/Partition");
        const float* centroid_a = partition.centroid_a.get();
        const float* centroid_b = partition.centroid_b.get();
        const size_t num_neighbors = neighboring_clusters_ids.size();
        std::unique_ptr<float[]> decoded_neighbor_centroids(new float[num_neighbors * d]);
        std::unique_ptr<const float*[]> neighbor_centroids(new const float*[num_neighbors]);
        // TODO(@lkuffo, low): We might consider avoiding decompression if we
        // compute the distances on the quantized domain
        for (size_t j = 0; j < num_neighbors; j++) {
            neighbor_centroids[j] = index.centroids.GetF32CentroidPtr(
                neighboring_clusters_ids[j], decoded_neighbor_centroids.get() + j * d
            );
        }
        for (size_t i = 0; i < num_embeddings; i++) {
            const float* emb = cluster_embeddings + i * d;
            float dist_old = distance_computer_f32_t::Horizontal(emb, centroid_to_split, d);
            // TODO(@lkuffo, low): We could avoid one of these
            // since we have the distance from k-means, we just need to bring it here
            float dist_a = distance_computer_f32_t::Horizontal(emb, centroid_a, d);
            float dist_b = distance_computer_f32_t::Horizontal(emb, centroid_b, d);
            float min_ab = std::min(dist_a, dist_b);
            auto& group_ab = dist_a <= dist_b ? partition.group_a_idx : partition.group_b_idx;
            if (min_ab <= dist_old) {
                group_ab.push_back(i);
                continue;
            }
            bool closer_elsewhere = false;
            for (size_t j = 0; j < num_neighbors; j++) {
                float dist = distance_computer_f32_t::Horizontal(emb, neighbor_centroids[j], d);
                if (dist < min_ab) {
                    closer_elsewhere = true;
                    break;
                }
            }
            if (closer_elsewhere) {
                partition.group_rest_idx.push_back(i);
            } else {
                group_ab.push_back(i);
            }
        }
    }
    return partition;
}

// Per neighboring cluster: (slot, goes to A) of each row to steal
using NeighborRowsToSteal = std::vector<std::vector<std::pair<uint32_t, bool>>>;

// The rows to steal that go to A and to B
inline std::pair<size_t, size_t> CountRowsToSteal(const NeighborRowsToSteal& rows_to_steal) {
    size_t num_rows = 0;
    size_t num_rows_to_a = 0;
    for (const auto& neighbor_rows : rows_to_steal) {
        num_rows += neighbor_rows.size();
        for (const auto& [slot, goes_to_a] : neighbor_rows) {
            num_rows_to_a += goes_to_a;
        }
    }
    return {num_rows_to_a, num_rows - num_rows_to_a};
}

// Keeps only the rows to steal that fit A and B under max_cluster_size (0: no limit); the rest
// stay in their own clusters
inline void CapRowsToSteal(
    NeighborRowsToSteal& rows_to_steal,
    uint32_t max_cluster_size,
    size_t size_a,
    size_t size_b
) {
    if (max_cluster_size == 0) {
        return;
    }
    size_t room_a = max_cluster_size > size_a ? max_cluster_size - size_a : 0;
    size_t room_b = max_cluster_size > size_b ? max_cluster_size - size_b : 0;
    for (auto& neighbor_rows : rows_to_steal) {
        size_t num_kept = 0;
        for (const auto& row : neighbor_rows) {
            size_t& room = row.second ? room_a : room_b;
            if (room > 0) {
                --room;
                neighbor_rows[num_kept++] = row;
            }
        }
        neighbor_rows.resize(num_kept);
    }
}

// For each neighboring cluster: the slots of its live rows whose own centroid is not strictly
// closer than both A and B, and whether each goes to A. One GEMM per neighbor, of its gathered
// (dequantized) rows against [own, A, B], in parallel over the neighbors.
template <Quantization Q>
inline NeighborRowsToSteal FindNeighborRowsToSteal(
    const IVF<Q>& index,
    const std::vector<uint32_t>& neighboring_clusters_ids,
    const float* centroid_a,
    const float* centroid_b,
    ParallelExecutor& executor,
    MemoryAllocator& allocator
) {
    PDX_PROFILE_SCOPE("Split/NeighborReassign");
    const size_t d = index.num_dimensions;
    const float norm_a = Eigen::Map<const Eigen::VectorXf>(centroid_a, d).squaredNorm();
    const float norm_b = Eigen::Map<const Eigen::VectorXf>(centroid_b, d).squaredNorm();
    NeighborRowsToSteal rows_to_steal(neighboring_clusters_ids.size());
    const ClusterFormat<Q> cluster_format = index.GetClusterFormat();
    executor.ParallelFor(neighboring_clusters_ids.size(), [&](size_t begin, size_t end, size_t) {
        ScalarQuantizer<Q> quantizer(d);
        const SAQQuantizer saq_quantizer(d);
        std::unique_ptr<pdx_data_t<Q>[]> raw_row(new pdx_data_t<Q>[index.GetValuesPerEmbedding()]);
        std::unique_ptr<float[]> centroids(new float[3 * d]);
        std::memcpy(centroids.get() + d, centroid_a, d * sizeof(float));
        std::memcpy(centroids.get() + 2 * d, centroid_b, d * sizeof(float));
        for (size_t j = begin; j < end; j++) {
            const uint32_t neighbor_id = neighboring_clusters_ids[j];
            const Cluster<Q>& neighbor = index.clusters[neighbor_id];
            const size_t n = neighbor.used_capacity;
            if (n == 0) {
                continue;
            }
            index.centroids.DecodeCentroid(neighbor_id, centroids.get());
            auto dots = skmeans::AllocateBuffer<float>(allocator, n * 3);
            if constexpr (Q == U8 || Q == SAQ) {
                auto rows = skmeans::AllocateBuffer<float>(allocator, n * d);
                for (size_t p = 0; p < n; p++) {
                    Cluster<Q>::ReadEmbeddingFromPDXBuffer(
                        neighbor.data.get(), neighbor.max_capacity, cluster_format, p, raw_row.get()
                    );
                    if constexpr (Q == SAQ) {
                        saq_quantizer.DecodeEmbedding(
                            raw_row.get(), index.saq_model, rows.get() + p * d
                        );
                    } else {
                        quantizer.DequantizeEmbedding(
                            raw_row.get(),
                            index.quantization_base,
                            index.quantization_scale,
                            rows.get() + p * d
                        );
                    }
                }
                skmeans::Sgemm(
                    'T',
                    'N',
                    3,
                    static_cast<int>(n),
                    static_cast<int>(d),
                    1.0f,
                    centroids.get(),
                    static_cast<int>(d),
                    rows.get(),
                    static_cast<int>(d),
                    0.0f,
                    dots.get(),
                    3
                );
            } else {
                // Straight on the PDX layout: the vertical block holds the rows transposed (one
                // dimension per row of stride max_capacity), each horizontal chunk is n x 64
                const size_t stride = neighbor.max_capacity;
                const size_t d_vertical = index.num_vertical_dimensions;
                skmeans::Sgemm(
                    'T',
                    'T',
                    3,
                    static_cast<int>(n),
                    static_cast<int>(d_vertical),
                    1.0f,
                    centroids.get(),
                    static_cast<int>(d),
                    neighbor.data.get(),
                    static_cast<int>(stride),
                    0.0f,
                    dots.get(),
                    3
                );
                for (size_t dim = d_vertical; dim < d; dim += H_DIM_SIZE) {
                    skmeans::Sgemm(
                        'T',
                        'N',
                        3,
                        static_cast<int>(n),
                        static_cast<int>(H_DIM_SIZE),
                        1.0f,
                        centroids.get() + dim,
                        static_cast<int>(d),
                        neighbor.data.get() + stride * dim,
                        static_cast<int>(H_DIM_SIZE),
                        1.0f,
                        dots.get(),
                        3
                    );
                }
            }
            // |x|^2 is common to the three distances of a row, so it is left out
            const float norm_own =
                Eigen::Map<const Eigen::VectorXf>(centroids.get(), d).squaredNorm();
            for (size_t p = 0; p < n; p++) {
                const auto slot = static_cast<uint32_t>(p);
                if (neighbor.HasTombstone(slot)) {
                    continue;
                }
                const float dist_own = norm_own - 2.0f * dots[p * 3];
                const float dist_a = norm_a - 2.0f * dots[p * 3 + 1];
                const float dist_b = norm_b - 2.0f * dots[p * 3 + 2];
                if (!(dist_own < dist_a && dist_own < dist_b)) {
                    rows_to_steal[j].emplace_back(slot, dist_a <= dist_b);
                }
            }
        }
    });
    return rows_to_steal;
}

// Steal from the neighboring clusters the rows of FindNeighborRowsToSteal: they are tombstoned in
// the neighbor and written to the A/B groups (raw embeddings, row ids; embs_a and ids_a have room
// for the rows that go to A, see CountRowsToSteal), their sum accumulated into the centroid sums.
// The caller must refresh the row-id mapping of the moved ids once the new clusters exist.
template <Quantization Q>
inline void StealNeighborEmbeddings(
    IVF<Q>& index,
    const std::vector<uint32_t>& neighboring_clusters_ids,
    const NeighborRowsToSteal& rows_to_steal,
    pdx_data_t<Q>* embs_a,
    uint32_t* ids_a,
    float* centroid_sum_a,
    pdx_data_t<Q>* embs_b,
    uint32_t* ids_b,
    float* centroid_sum_b,
    uint8_t* sketches_a = nullptr,
    uint8_t* sketches_b = nullptr
) {
    PDX_PROFILE_SCOPE("Split/NeighborReassign");
    const size_t d = index.num_dimensions;
    const size_t values_per_embedding = index.GetValuesPerEmbedding();
    const size_t sketch_row_size = ClusterSketchQuantizer(index.num_dimensions).row_size_in_bytes;
    ScalarQuantizer<Q> quantizer(d);
    const SAQQuantizer saq_quantizer(d);
    std::unique_ptr<float[]> emb_f32(new float[d]);
    size_t num_rows_to_a = 0;
    size_t num_rows_to_b = 0;
    for (size_t j = 0; j < neighboring_clusters_ids.size(); j++) {
        auto& neighbor = index.clusters[neighboring_clusters_ids[j]];
        for (const auto& [slot, goes_to_a] : rows_to_steal[j]) {
            auto raw_emb = neighbor.GetHorizontalEmbeddingFromPDXBuffer(slot);
            const float* emb_ptr;
            if constexpr (Q == SAQ) {
                saq_quantizer.DecodeEmbedding(raw_emb.get(), index.saq_model, emb_f32.get());
                emb_ptr = emb_f32.get();
            } else if constexpr (Q == U8) {
                quantizer.DequantizeEmbedding(
                    raw_emb.get(), index.quantization_base, index.quantization_scale, emb_f32.get()
                );
                emb_ptr = emb_f32.get();
            } else {
                emb_ptr = raw_emb.get();
            }

            const uint32_t row_id = neighbor.indices[slot];
            const size_t position = goes_to_a ? num_rows_to_a++ : num_rows_to_b++;
            // Before DeleteEmbedding clears the lane
            if (sketches_a != nullptr) {
                neighbor.ReadSketchRow(
                    slot, (goes_to_a ? sketches_a : sketches_b) + position * sketch_row_size
                );
            }
            neighbor.DeleteEmbedding(slot);
            pdx_data_t<Q>* embs = goes_to_a ? embs_a : embs_b;
            uint32_t* ids = goes_to_a ? ids_a : ids_b;
            float* centroid_sum = goes_to_a ? centroid_sum_a : centroid_sum_b;
            std::memcpy(
                embs + position * values_per_embedding,
                raw_emb.get(),
                values_per_embedding * sizeof(pdx_data_t<Q>)
            );
            ids[position] = row_id;
            for (size_t k = 0; k < d; k++) {
                centroid_sum[k] += emb_ptr[k];
            }
        }
    }
}

} // namespace PDX