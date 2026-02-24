#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

#include "pdx/clustering.hpp"
#include "pdx/common.hpp"
#include "pdx/ivf_wrapper.hpp"
#include "pdx/layout.hpp"
#include "pdx/pruners/adsampling.hpp"
#include "pdx/quantizers/scalar.hpp"
#include "pdx/searcher.hpp"
#include "pdx/utils.hpp"

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
};

class IPDXIndex {
  public:
    virtual ~IPDXIndex() = default;
    virtual std::vector<KNNCandidate> Search(const float* query_embedding, size_t knn) const = 0;
    virtual void BuildIndex(const float* embeddings, size_t num_embeddings) = 0;
    virtual void SetNProbe(uint32_t n_probe) const = 0;
    virtual void Save(const std::string& path) const = 0;
    virtual void Restore(const std::string& path) = 0;
    virtual uint32_t GetNumDimensions() const = 0;
    virtual uint32_t GetNumClusters() const = 0;
    virtual size_t GetInMemorySizeInBytes() const = 0;
};

template <PDX::Quantization Q>
class PDXIndex : public IPDXIndex {
  public:
    using embedding_storage_t = PDX::pdx_data_t<Q>;

  private:
    PDXIndexConfig config{};
    PDX::IVF<Q> index;
    std::unique_ptr<PDX::ADSamplingPruner> pruner;
    std::unique_ptr<PDX::PDXearch<Q>> searcher;

    static constexpr PDXIndexType GetIndexType() {
        if constexpr (Q == F32)
            return PDXIndexType::PDX_F32;
        else
            return PDXIndexType::PDX_U8;
    }

  public:
    PDXIndex() = default;

    explicit PDXIndex(PDXIndexConfig config) : config(config) {
        pruner = std::make_unique<PDX::ADSamplingPruner>(config.num_dimensions, config.seed);
    }

    void Save(const std::string& path) const override {
        std::ofstream out(path, std::ios::binary);

        // Index type flag
        uint8_t type_flag = static_cast<uint8_t>(GetIndexType());
        out.write(reinterpret_cast<const char*>(&type_flag), sizeof(uint8_t));

        // Rotation matrix
        const auto& matrix = pruner->GetMatrix();
        uint32_t matrix_rows = static_cast<uint32_t>(matrix.rows());
        uint32_t matrix_cols = static_cast<uint32_t>(matrix.cols());
        out.write(reinterpret_cast<const char*>(&matrix_rows), sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(&matrix_cols), sizeof(uint32_t));
        out.write(
            reinterpret_cast<const char*>(matrix.data()), sizeof(float) * matrix_rows * matrix_cols
        );

        // IVF data
        index.Save(out);
    }

    void Restore(const std::string& path) override {
        auto buffer = MmapFile(path);
        char* ptr = buffer.get();

        // Index type flag
        ptr += sizeof(uint8_t);

        // Rotation matrix
        uint32_t matrix_rows = *reinterpret_cast<uint32_t*>(ptr);
        ptr += sizeof(uint32_t);
        uint32_t matrix_cols = *reinterpret_cast<uint32_t*>(ptr);
        ptr += sizeof(uint32_t);
        auto* matrix_data = reinterpret_cast<float*>(ptr);
        ptr += sizeof(float) * matrix_rows * matrix_cols;

        // Load IVF data
        index.Load(ptr);

        // Create pruner and searcher
        pruner = std::make_unique<PDX::ADSamplingPruner>(index.num_dimensions, matrix_data);
        searcher = std::make_unique<PDX::PDXearch<Q>>(index, *pruner);
    }

    std::vector<PDX::KNNCandidate> Search(const float* query_embedding, size_t knn) const override {
        return searcher->Search(query_embedding, knn);
    }

    void SetNProbe(uint32_t n_probe) const override { searcher->SetNProbe(n_probe); }

    const PDX::PDXearch<Q>& GetSearcher() const { return *searcher; }

    uint32_t GetNumDimensions() const override { return index.num_dimensions; }

    uint32_t GetNumClusters() const override { return index.num_clusters; }

    size_t GetInMemorySizeInBytes() const override {
        size_t size = sizeof(*this);
        // IVF heap allocations (sizeof(IVF<Q>) is inline in sizeof(*this))
        size += index.GetInMemorySizeInBytes() - sizeof(index);
        // Pruner: rotation matrix (D x D floats) + ratios vector (D floats)
        if (pruner) {
            size += sizeof(*pruner);
            const auto& m = pruner->GetMatrix();
            size += static_cast<size_t>(m.rows()) * m.cols() * sizeof(float);
            size += pruner->num_dimensions * sizeof(float);
        }
        // Searcher: cluster_offsets array
        if (searcher) {
            size += sizeof(*searcher);
            size += index.num_clusters * sizeof(size_t);
        }
        return size;
    }

    void BuildIndex(const float* const embeddings, const size_t num_embeddings) override {
        std::vector<size_t> row_ids(num_embeddings);
        std::iota(row_ids.begin(), row_ids.end(), 0);
        BuildIndex(row_ids.data(), embeddings, num_embeddings);
    }

    void BuildIndex(
        const size_t* const row_ids,
        const float* const embeddings,
        const size_t num_embeddings
    ) {
        const auto num_dimensions = config.num_dimensions;
        auto num_clusters = config.num_clusters;
        if (num_clusters == 0) {
            num_clusters = ComputeNumberOfClusters(num_embeddings);
        }
        const bool normalize =
            config.normalize || DistanceMetricRequiresNormalization(config.distance_metric);

        assert(num_dimensions > 0);
        assert(num_embeddings > 0);
        assert(pruner);

        // Preprocess: normalize (if needed) + rotate.
        const size_t total_floats = num_embeddings * num_dimensions;
        std::unique_ptr<float[]> normalized;
        const float* rotation_input = embeddings;
        if (normalize) {
            normalized.reset(new float[total_floats]);
            std::memcpy(normalized.get(), embeddings, total_floats * sizeof(float));
            PDX::Quantizer quantizer(num_dimensions);
            for (size_t i = 0; i < num_embeddings; i++) {
                quantizer.NormalizeQuery(
                    normalized.get() + i * num_dimensions, normalized.get() + i * num_dimensions
                );
            }
            rotation_input = normalized.get();
        }
        std::unique_ptr<float[]> preprocessed(new float[total_floats]);
        pruner->PreprocessEmbeddings(rotation_input, preprocessed.get(), num_embeddings);

        float quantization_base = 0.0f;
        float quantization_scale = 1.0f;
        if constexpr (Q == PDX::U8) {
            const auto params = PDX::ScalarQuantizer<Q>::ComputeQuantizationParams(
                preprocessed.get(), static_cast<size_t>(num_embeddings) * num_dimensions
            );
            quantization_base = params.quantization_base;
            quantization_scale = params.quantization_scale;
            index = PDX::IVF<Q>(
                num_dimensions,
                num_embeddings,
                num_clusters,
                normalize,
                quantization_scale,
                quantization_base
            );
        } else {
            index = PDX::IVF<Q>(num_dimensions, num_embeddings, num_clusters, normalize);
        }

        KMeansResult kmeans_result = ComputeKMeans(
            preprocessed.get(),
            num_embeddings,
            num_dimensions,
            num_clusters,
            config.distance_metric,
            config.seed,
            config.normalize,
            config.sampling_fraction,
            config.kmeans_iters
        );
        index.centroids = std::move(kmeans_result.centroids);

        size_t max_cluster_size = 0;
        for (size_t i = 0; i < num_clusters; i++) {
            max_cluster_size = std::max(max_cluster_size, kmeans_result.assignments[i].size());
        }
        std::unique_ptr<embedding_storage_t[]> tmp_cluster_embeddings(
            new embedding_storage_t[static_cast<uint64_t>(max_cluster_size) * num_dimensions]
        );

        for (size_t cluster_idx = 0; cluster_idx < num_clusters; cluster_idx++) {
            const auto cluster_size = kmeans_result.assignments[cluster_idx].size();
            auto& cluster = index.clusters.emplace_back(cluster_size, num_dimensions);

            for (size_t position_in_cluster = 0; position_in_cluster < cluster_size;
                 position_in_cluster++) {
                const auto embedding_idx =
                    kmeans_result.assignments[cluster_idx][position_in_cluster];
                cluster.indices[position_in_cluster] = row_ids[embedding_idx];

                if constexpr (Q == PDX::U8) {
                    PDX::ScalarQuantizer<Q> quantizer(num_dimensions);
                    quantizer.QuantizeEmbedding(
                        preprocessed.get() + (embedding_idx * num_dimensions),
                        quantization_base,
                        quantization_scale,
                        tmp_cluster_embeddings.get() + (position_in_cluster * num_dimensions)
                    );
                } else {
                    std::memcpy(
                        tmp_cluster_embeddings.get() + (position_in_cluster * num_dimensions),
                        preprocessed.get() + (embedding_idx * num_dimensions),
                        num_dimensions * sizeof(float)
                    );
                }
            }
            StoreClusterEmbeddings<Q, embedding_storage_t>(
                cluster, index, tmp_cluster_embeddings.get(), cluster_size
            );
        }

        searcher = std::make_unique<PDX::PDXearch<Q>>(index, *pruner);
    }
};

template <PDX::Quantization Q>
class PDXTreeIndex : public IPDXIndex {
  public:
    using embedding_storage_t = PDX::pdx_data_t<Q>;

  private:
    PDXIndexConfig config{};
    PDX::IVFTree<Q> index;
    std::unique_ptr<PDX::ADSamplingPruner> pruner;
    std::unique_ptr<PDX::PDXearch<Q>> searcher;
    std::unique_ptr<PDX::PDXearch<F32>> top_level_searcher;

    static constexpr PDXIndexType GetIndexType() {
        if constexpr (Q == F32)
            return PDXIndexType::PDX_TREE_F32;
        else
            return PDXIndexType::PDX_TREE_U8;
    }

  public:
    PDXTreeIndex() = default;

    explicit PDXTreeIndex(PDXIndexConfig config) : config(config) {
        pruner = std::make_unique<PDX::ADSamplingPruner>(config.num_dimensions, config.seed);
    }

    void Save(const std::string& path) const override {
        std::ofstream out(path, std::ios::binary);

        // Index type flag
        uint8_t type_flag = static_cast<uint8_t>(GetIndexType());
        out.write(reinterpret_cast<const char*>(&type_flag), sizeof(uint8_t));

        // Rotation matrix
        const auto& matrix = pruner->GetMatrix();
        uint32_t matrix_rows = static_cast<uint32_t>(matrix.rows());
        uint32_t matrix_cols = static_cast<uint32_t>(matrix.cols());
        out.write(reinterpret_cast<const char*>(&matrix_rows), sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(&matrix_cols), sizeof(uint32_t));
        out.write(
            reinterpret_cast<const char*>(matrix.data()), sizeof(float) * matrix_rows * matrix_cols
        );

        // IVFTree data
        index.Save(out);
    }

    void Restore(const std::string& path) override {
        auto buffer = MmapFile(path);
        char* ptr = buffer.get();

        // Index type flag
        ptr += sizeof(uint8_t);

        // Rotation matrix
        uint32_t matrix_rows = *reinterpret_cast<uint32_t*>(ptr);
        ptr += sizeof(uint32_t);
        uint32_t matrix_cols = *reinterpret_cast<uint32_t*>(ptr);
        ptr += sizeof(uint32_t);
        auto* matrix_data = reinterpret_cast<float*>(ptr);
        ptr += sizeof(float) * matrix_rows * matrix_cols;

        // Load IVFTree data
        index.Load(ptr);

        // Create pruner and searchers
        pruner = std::make_unique<PDX::ADSamplingPruner>(index.num_dimensions, matrix_data);
        searcher = std::make_unique<PDX::PDXearch<Q>>(index, *pruner);
        top_level_searcher = std::make_unique<PDX::PDXearch<F32>>(index.l0, *pruner);
    }

    std::vector<PDX::KNNCandidate> Search(const float* query_embedding, size_t knn) const override {
        auto n_probe_top_level = GetTopLevelNumClusters();
        // We confidently prune half of the search space
        if (searcher->GetNProbe() < GetNumClusters() / 2) {
            n_probe_top_level /= 2;
        }
        top_level_searcher->SetNProbe(n_probe_top_level);
        auto top_level_results = top_level_searcher->Search(query_embedding, searcher->GetNProbe());

        std::vector<uint32_t> top_level_indexes(top_level_results.size());
        for (size_t i = 0; i < top_level_results.size(); i++) {
            top_level_indexes[i] = top_level_results[i].index;
        }
        searcher->SetClusterAccessOrder(top_level_indexes);

        return searcher->Search(query_embedding, knn);
    }

    void BuildIndex(const float* const embeddings, const size_t num_embeddings) override {
        std::vector<size_t> row_ids(num_embeddings);
        std::iota(row_ids.begin(), row_ids.end(), 0);
        BuildIndex(row_ids.data(), embeddings, num_embeddings);
    }

    void BuildIndex(
        const size_t* const row_ids,
        const float* const embeddings,
        const size_t num_embeddings
    ) {
        const auto num_dimensions = config.num_dimensions;
        auto num_clusters = config.num_clusters;
        if (num_clusters == 0) {
            num_clusters = ComputeNumberOfClusters(num_embeddings);
        }
        const bool normalize =
            config.normalize || DistanceMetricRequiresNormalization(config.distance_metric);

        assert(num_dimensions > 0);
        assert(num_embeddings > 0);
        assert(pruner);

        // Preprocess: normalize (if needed) + rotate.
        const size_t total_floats = num_embeddings * num_dimensions;
        std::unique_ptr<float[]> normalized;
        const float* rotation_input = embeddings;
        if (normalize) {
            normalized.reset(new float[total_floats]);
            std::memcpy(normalized.get(), embeddings, total_floats * sizeof(float));
            PDX::Quantizer quantizer(num_dimensions);
            for (size_t i = 0; i < num_embeddings; i++) {
                quantizer.NormalizeQuery(
                    normalized.get() + i * num_dimensions, normalized.get() + i * num_dimensions
                );
            }
            rotation_input = normalized.get();
        }
        std::unique_ptr<float[]> preprocessed(new float[total_floats]);
        pruner->PreprocessEmbeddings(rotation_input, preprocessed.get(), num_embeddings);

        float quantization_base = 0.0f;
        float quantization_scale = 1.0f;
        if constexpr (Q == PDX::U8) {
            const auto params = PDX::ScalarQuantizer<Q>::ComputeQuantizationParams(
                preprocessed.get(), static_cast<size_t>(num_embeddings) * num_dimensions
            );
            quantization_base = params.quantization_base;
            quantization_scale = params.quantization_scale;
            index = PDX::IVFTree<Q>(
                num_dimensions,
                num_embeddings,
                num_clusters,
                normalize,
                quantization_scale,
                quantization_base
            );
        } else {
            index = PDX::IVFTree<Q>(num_dimensions, num_embeddings, num_clusters, normalize);
        }

        KMeansResult kmeans_result = ComputeKMeans(
            preprocessed.get(),
            num_embeddings,
            num_dimensions,
            num_clusters,
            config.distance_metric,
            config.seed,
            config.normalize,
            config.sampling_fraction,
            config.kmeans_iters
        );
        index.centroids = std::move(kmeans_result.centroids);

        size_t max_cluster_size = 0;
        for (size_t i = 0; i < num_clusters; i++) {
            max_cluster_size = std::max(max_cluster_size, kmeans_result.assignments[i].size());
        }
        std::unique_ptr<embedding_storage_t[]> tmp_cluster_embeddings(
            new embedding_storage_t[static_cast<uint64_t>(max_cluster_size) * num_dimensions]
        );

        for (size_t cluster_idx = 0; cluster_idx < num_clusters; cluster_idx++) {
            const auto cluster_size = kmeans_result.assignments[cluster_idx].size();
            auto& cluster = index.clusters.emplace_back(cluster_size, num_dimensions);

            for (size_t position_in_cluster = 0; position_in_cluster < cluster_size;
                 position_in_cluster++) {
                const auto embedding_idx =
                    kmeans_result.assignments[cluster_idx][position_in_cluster];
                cluster.indices[position_in_cluster] = row_ids[embedding_idx];

                if constexpr (Q == PDX::U8) {
                    PDX::ScalarQuantizer<Q> quantizer(num_dimensions);
                    quantizer.QuantizeEmbedding(
                        preprocessed.get() + (embedding_idx * num_dimensions),
                        quantization_base,
                        quantization_scale,
                        tmp_cluster_embeddings.get() + (position_in_cluster * num_dimensions)
                    );
                } else {
                    std::memcpy(
                        tmp_cluster_embeddings.get() + (position_in_cluster * num_dimensions),
                        preprocessed.get() + (embedding_idx * num_dimensions),
                        num_dimensions * sizeof(float)
                    );
                }
            }
            StoreClusterEmbeddings<Q, embedding_storage_t>(
                cluster, index, tmp_cluster_embeddings.get(), cluster_size
            );
        }
        searcher = std::make_unique<PDX::PDXearch<Q>>(index, *pruner);

        // L0 index
        auto l0_num_clusters = config.num_meso_clusters;
        if (l0_num_clusters == 0) {
            l0_num_clusters = static_cast<uint32_t>(std::sqrt(num_clusters));
        }

        index.l0 = PDX::IVF<F32>(num_dimensions, num_clusters, l0_num_clusters, normalize);
        KMeansResult l0_kmeans_result = ComputeKMeans(
            index.centroids.data(),
            num_clusters,
            num_dimensions,
            l0_num_clusters,
            config.distance_metric,
            config.seed,
            config.normalize,
            1.0f,
            10
        ); // No sampling for l0
        index.l0.centroids = std::move(l0_kmeans_result.centroids);

        size_t l0_max_cluster_size = 0;
        for (size_t i = 0; i < l0_num_clusters; i++) {
            l0_max_cluster_size =
                std::max(l0_max_cluster_size, l0_kmeans_result.assignments[i].size());
        }
        std::unique_ptr<float[]> l0_tmp_cluster_embeddings(
            new float[static_cast<uint64_t>(l0_max_cluster_size) * num_dimensions]
        );

        for (size_t cluster_idx = 0; cluster_idx < l0_num_clusters; cluster_idx++) {
            const auto cluster_size = l0_kmeans_result.assignments[cluster_idx].size();
            auto& cluster = index.l0.clusters.emplace_back(cluster_size, num_dimensions);

            for (size_t position_in_cluster = 0; position_in_cluster < cluster_size;
                 position_in_cluster++) {
                const auto embedding_idx =
                    l0_kmeans_result.assignments[cluster_idx][position_in_cluster];
                cluster.indices[position_in_cluster] = embedding_idx;

                std::memcpy(
                    l0_tmp_cluster_embeddings.get() + (position_in_cluster * num_dimensions),
                    index.centroids.data() + (embedding_idx * num_dimensions),
                    num_dimensions * sizeof(float)
                );
            }
            StoreClusterEmbeddings<F32, float>(
                cluster, index.l0, l0_tmp_cluster_embeddings.get(), cluster_size
            );
        }

        top_level_searcher = std::make_unique<PDX::PDXearch<F32>>(index.l0, *pruner);
    }

    void SetNProbe(uint32_t n_probe) const override { searcher->SetNProbe(n_probe); }

    const PDX::PDXearch<Q>& GetSearcher() const { return *searcher; }

    uint32_t GetNumDimensions() const override { return index.num_dimensions; }

    uint32_t GetNumClusters() const override { return index.num_clusters; }

    uint32_t GetTopLevelNumClusters() const { return index.l0.num_clusters; }

    size_t GetInMemorySizeInBytes() const override {
        size_t size = sizeof(*this);
        // IVFTree heap allocations (L1 + L0 clusters and centroids)
        size += index.GetInMemorySizeInBytes() - sizeof(index);
        // Pruner: rotation matrix (D x D floats) + ratios vector (D floats)
        if (pruner) {
            size += sizeof(*pruner);
            const auto& m = pruner->GetMatrix();
            size += static_cast<size_t>(m.rows()) * m.cols() * sizeof(float);
            size += pruner->num_dimensions * sizeof(float);
        }
        // L1 searcher: cluster_offsets array
        if (searcher) {
            size += sizeof(*searcher);
            size += index.num_clusters * sizeof(size_t);
        }
        // L0 top-level searcher: cluster_offsets array
        if (top_level_searcher) {
            size += sizeof(*top_level_searcher);
            size += index.l0.num_clusters * sizeof(size_t);
        }
        return size;
    }
};

using PDXIndexF32 = PDXIndex<PDX::F32>;
using PDXIndexU8 = PDXIndex<PDX::U8>;
using PDXTreeIndexF32 = PDXTreeIndex<PDX::F32>;
using PDXTreeIndexU8 = PDXTreeIndex<PDX::U8>;

inline std::unique_ptr<IPDXIndex> LoadPDXIndex(const std::string& path) {
    auto buffer = MmapFile(path);
    auto type = static_cast<PDXIndexType>(buffer.get()[0]);
    std::unique_ptr<IPDXIndex> idx;
    switch (type) {
    case PDXIndexType::PDX_F32:
        idx = std::make_unique<PDXIndexF32>();
        break;
    case PDXIndexType::PDX_U8:
        idx = std::make_unique<PDXIndexU8>();
        break;
    case PDXIndexType::PDX_TREE_F32:
        idx = std::make_unique<PDXTreeIndexF32>();
        break;
    case PDXIndexType::PDX_TREE_U8:
        idx = std::make_unique<PDXTreeIndexU8>();
        break;
    default:
        throw std::runtime_error(
            "Unknown PDX index type: " + std::to_string(static_cast<int>(type))
        );
    }
    idx->Restore(path);
    return idx;
}

class EmbeddingPreprocessor {
  private:
    PDX::ADSamplingPruner pruner;
    PDX::Quantizer quantizer;
    const size_t num_dimensions;

  public:
    explicit EmbeddingPreprocessor(const size_t num_dimensions, const float* const rotation_matrix)
        : pruner(num_dimensions, rotation_matrix), quantizer(num_dimensions),
          num_dimensions(num_dimensions) {}

    void PreprocessEmbedding(
        float* const input_embedding,
        float* const output_embedding,
        const bool normalize
    ) const {
        if (normalize) {
            quantizer.NormalizeQuery(input_embedding, input_embedding);
        }
        pruner.PreprocessQuery(input_embedding, output_embedding);
    }

    void PreprocessEmbeddings(
        float* const input_embeddings,
        float* const output_embeddings,
        const size_t num_embeddings,
        const bool normalize
    ) const {
        if (normalize) {
            for (size_t i = 0; i < num_embeddings; i++) {
                quantizer.NormalizeQuery(
                    input_embeddings + i * num_dimensions, input_embeddings + i * num_dimensions
                );
            }
        }
        pruner.PreprocessEmbeddings(input_embeddings, output_embeddings, num_embeddings);
    }
};

} // namespace PDX
