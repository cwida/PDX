#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>

#include "common.hpp"
#include "ivf_wrapper.hpp"
#include "pruners/adsampling.hpp"
#include "pdxearch.hpp"
#include "utils/file_reader.hpp"
#include "clustering.hpp"
#include "utils/utils.hpp"
#include "quantizers/scalar.hpp"

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

template <PDX::Quantization Q>
class PDXIndex {
public:
	using embedding_storage_t = PDX::pdx_data_t<Q>;

private:
	std::unique_ptr<char[]> matrix_buffer;

	PDXIndexConfig config {};
	PDX::IndexPDXIVF<Q> index;
	std::unique_ptr<PDX::ADSamplingPruner> pruner;
	std::unique_ptr<PDX::PDXearch<Q>> searcher;

public:
	PDXIndex() = default;

	explicit PDXIndex(PDXIndexConfig config) : config(config) {
		pruner = std::make_unique<PDX::ADSamplingPruner>(config.num_dimensions, config.seed);
	}

	void Restore(const std::string &index_path, const std::string &matrix_path) {
		index.Restore(index_path);

		matrix_buffer = MmapFile(matrix_path);
		auto *matrix = reinterpret_cast<float *>(matrix_buffer.get());

		pruner = std::make_unique<PDX::ADSamplingPruner>(index.num_dimensions, matrix);
		searcher = std::make_unique<PDX::PDXearch<Q>>(index, *pruner);
	}

	std::vector<PDX::KNNCandidate> Search(const float *query_embedding, size_t knn) const {
		return searcher->Search(query_embedding, knn);
	}

	void SetNProbe(uint32_t n_probe) const {
		searcher->SetNProbe(n_probe);
	}

	const PDX::PDXearch<Q> &GetSearcher() const {
		return *searcher;
	}

	uint32_t GetNumDimensions() const {
		return index.num_dimensions;
	}

	uint32_t GetNumClusters() const {
		return index.num_clusters;
	}

	void BuildIndex(const float *const embeddings, const size_t num_embeddings) {
		std::vector<size_t> row_ids(num_embeddings);
		std::iota(row_ids.begin(), row_ids.end(), 0);
		BuildIndex(row_ids.data(), embeddings, num_embeddings);
	}

	void BuildIndex(const size_t *const row_ids, const float *const embeddings, const size_t num_embeddings) {
		const auto num_dimensions = config.num_dimensions;
		auto num_clusters = config.num_clusters;
		if (num_clusters == 0) {
			num_clusters = ComputeNumberOfClusters(num_embeddings);
		}
		const bool normalize = config.normalize || DistanceMetricRequiresNormalization(config.distance_metric);

		assert(num_dimensions > 0);
		assert(num_embeddings > 0);
		assert(pruner);

		// Preprocess: normalize (if needed) + rotate.
		const size_t total_floats = num_embeddings * num_dimensions;
		std::unique_ptr<float[]> normalized;
		const float *rotation_input = embeddings;
		if (normalize) {
			normalized.reset(new float[total_floats]);
			std::memcpy(normalized.get(), embeddings, total_floats * sizeof(float));
			PDX::Quantizer quantizer(num_dimensions);
			for (size_t i = 0; i < num_embeddings; i++) {
				quantizer.NormalizeQuery(normalized.get() + i * num_dimensions,
				                         normalized.get() + i * num_dimensions);
			}
			rotation_input = normalized.get();
		}
		std::unique_ptr<float[]> preprocessed(new float[total_floats]);
		pruner->PreprocessEmbeddings(rotation_input, preprocessed.get(), num_embeddings);

		float quantization_base = 0.0f;
		float quantization_scale = 1.0f;
		if constexpr (Q == PDX::U8) {
			const auto params = PDX::ScalarQuantizer<Q>::ComputeQuantizationParams(
			    preprocessed.get(), static_cast<size_t>(num_embeddings) * num_dimensions);
			quantization_base = params.quantization_base;
			quantization_scale = params.quantization_scale;
			index = PDX::IndexPDXIVF<Q>(num_dimensions, num_embeddings, num_clusters,
			                             normalize, quantization_scale, quantization_base);
		} else {
			index = PDX::IndexPDXIVF<Q>(num_dimensions, num_embeddings, num_clusters, normalize);
		}

		KMeansResult kmeans_result = ComputeKMeans(preprocessed.get(), num_embeddings, num_dimensions,
		                                           num_clusters, config.distance_metric, config.seed,
		                                           config.normalize, config.sampling_fraction, config.kmeans_iters);
		index.centroids = std::move(kmeans_result.centroids);

		size_t max_cluster_size = 0;
		for (size_t i = 0; i < num_clusters; i++) {
			max_cluster_size = std::max(max_cluster_size, kmeans_result.assignments[i].size());
		}
		std::unique_ptr<embedding_storage_t[]> tmp_cluster_embeddings(
		    new embedding_storage_t[static_cast<uint64_t>(max_cluster_size) * num_dimensions]);

		for (size_t cluster_idx = 0; cluster_idx < num_clusters; cluster_idx++) {
			const auto cluster_size = kmeans_result.assignments[cluster_idx].size();
			auto &cluster = index.clusters.emplace_back(cluster_size, num_dimensions);

			for (size_t position_in_cluster = 0; position_in_cluster < cluster_size; position_in_cluster++) {
				const auto embedding_idx = kmeans_result.assignments[cluster_idx][position_in_cluster];
				cluster.indices[position_in_cluster] = row_ids[embedding_idx];

				if constexpr (Q == PDX::U8) {
					PDX::ScalarQuantizer<Q> quantizer(num_dimensions);
					quantizer.QuantizeEmbedding(preprocessed.get() + (embedding_idx * num_dimensions),
					                            quantization_base, quantization_scale,
					                            tmp_cluster_embeddings.get() + (position_in_cluster * num_dimensions));
				} else {
					std::memcpy(tmp_cluster_embeddings.get() + (position_in_cluster * num_dimensions),
					            preprocessed.get() + (embedding_idx * num_dimensions),
					            num_dimensions * sizeof(float));
				}
			}
			StoreClusterEmbeddings<Q, embedding_storage_t>(cluster, index, tmp_cluster_embeddings.get(),
			                                               cluster_size);
		}

		searcher = std::make_unique<PDX::PDXearch<Q>>(index, *pruner);
	}

};

template <PDX::Quantization Q>
class PDXTreeIndex {
public:
	using embedding_storage_t = PDX::pdx_data_t<Q>;

private:
	std::unique_ptr<char[]> matrix_buffer;

	PDXIndexConfig config {};
	PDX::IndexPDXIVF2<Q> index;
	std::unique_ptr<PDX::ADSamplingPruner> pruner;
	std::unique_ptr<PDX::PDXearch<Q>> searcher;
	std::unique_ptr<PDX::PDXearch<F32>> top_level_searcher;

public:
	PDXTreeIndex() = default;

	explicit PDXTreeIndex(PDXIndexConfig config) : config(config) {
		pruner = std::make_unique<PDX::ADSamplingPruner>(config.num_dimensions, config.seed);
	}

	void Restore(const std::string &index_path, const std::string &matrix_path) {
		index.Restore(index_path);

		matrix_buffer = MmapFile(matrix_path);
		auto *matrix = reinterpret_cast<float *>(matrix_buffer.get());

		pruner = std::make_unique<PDX::ADSamplingPruner>(index.num_dimensions, matrix);
		searcher = std::make_unique<PDX::PDXearch<Q>>(index, *pruner);
		top_level_searcher = std::make_unique<PDX::PDXearch<F32>>(index.l0, *pruner);
	}

	std::vector<PDX::KNNCandidate> Search(const float *query_embedding, size_t knn) const {
		auto n_probe_top_level = GetTopLevelNumClusters();
		// if (searcher->GetNProbe() < GetNumClusters() / 2){
		// 	// We confidently prune half of the meso-clusters only if the user wants to
        //     // visit less than half of the available clusters
		// 	n_probe_top_level /= 2;
		// }
		top_level_searcher->SetNProbe(n_probe_top_level);
		auto top_level_results = top_level_searcher->Search(query_embedding, searcher->GetNProbe());

		std::vector<uint32_t> top_level_indexes(top_level_results.size());
		for (size_t i = 0; i < top_level_results.size(); i++) {
			top_level_indexes[i] = top_level_results[i].index;
		}
		searcher->SetClusterAccessOrder(top_level_indexes);

		return searcher->Search(query_embedding, knn);
	}

	void BuildIndex(const float *const embeddings, const size_t num_embeddings) {
		std::vector<size_t> row_ids(num_embeddings);
		std::iota(row_ids.begin(), row_ids.end(), 0);
		BuildIndex(row_ids.data(), embeddings, num_embeddings);
	}

	void BuildIndex(const size_t *const row_ids, const float *const embeddings, const size_t num_embeddings) {
		const auto num_dimensions = config.num_dimensions;
		auto num_clusters = config.num_clusters;
		if (num_clusters == 0) {
			num_clusters = ComputeNumberOfClusters(num_embeddings);
		}
		const bool normalize = config.normalize || DistanceMetricRequiresNormalization(config.distance_metric);

		assert(num_dimensions > 0);
		assert(num_embeddings > 0);
		assert(pruner);

		// Preprocess: normalize (if needed) + rotate.
		const size_t total_floats = num_embeddings * num_dimensions;
		std::unique_ptr<float[]> normalized;
		const float *rotation_input = embeddings;
		if (normalize) {
			normalized.reset(new float[total_floats]);
			std::memcpy(normalized.get(), embeddings, total_floats * sizeof(float));
			PDX::Quantizer quantizer(num_dimensions);
			for (size_t i = 0; i < num_embeddings; i++) {
				quantizer.NormalizeQuery(normalized.get() + i * num_dimensions,
				                         normalized.get() + i * num_dimensions);
			}
			rotation_input = normalized.get();
		}
		std::unique_ptr<float[]> preprocessed(new float[total_floats]);
		pruner->PreprocessEmbeddings(rotation_input, preprocessed.get(), num_embeddings);

		float quantization_base = 0.0f;
		float quantization_scale = 1.0f;
		if constexpr (Q == PDX::U8) {
			const auto params = PDX::ScalarQuantizer<Q>::ComputeQuantizationParams(
			    preprocessed.get(), static_cast<size_t>(num_embeddings) * num_dimensions);
			quantization_base = params.quantization_base;
			quantization_scale = params.quantization_scale;
			index = PDX::IndexPDXIVF2<Q>(num_dimensions, num_embeddings, num_clusters,
			                             normalize, quantization_scale, quantization_base);
		} else {
			index = PDX::IndexPDXIVF2<Q>(num_dimensions, num_embeddings, num_clusters, normalize);
		}

		KMeansResult kmeans_result = ComputeKMeans(preprocessed.get(), num_embeddings, num_dimensions,
		                                           num_clusters, config.distance_metric, config.seed,
		                                           config.normalize, config.sampling_fraction, config.kmeans_iters);
		index.centroids = std::move(kmeans_result.centroids);

		size_t max_cluster_size = 0;
		for (size_t i = 0; i < num_clusters; i++) {
			max_cluster_size = std::max(max_cluster_size, kmeans_result.assignments[i].size());
		}
		std::unique_ptr<embedding_storage_t[]> tmp_cluster_embeddings(
		    new embedding_storage_t[static_cast<uint64_t>(max_cluster_size) * num_dimensions]);

		for (size_t cluster_idx = 0; cluster_idx < num_clusters; cluster_idx++) {
			const auto cluster_size = kmeans_result.assignments[cluster_idx].size();
			auto &cluster = index.clusters.emplace_back(cluster_size, num_dimensions);

			for (size_t position_in_cluster = 0; position_in_cluster < cluster_size; position_in_cluster++) {
				const auto embedding_idx = kmeans_result.assignments[cluster_idx][position_in_cluster];
				cluster.indices[position_in_cluster] = row_ids[embedding_idx];

				if constexpr (Q == PDX::U8) {
					PDX::ScalarQuantizer<Q> quantizer(num_dimensions);
					quantizer.QuantizeEmbedding(preprocessed.get() + (embedding_idx * num_dimensions),
					                            quantization_base, quantization_scale,
					                            tmp_cluster_embeddings.get() + (position_in_cluster * num_dimensions));
				} else {
					std::memcpy(tmp_cluster_embeddings.get() + (position_in_cluster * num_dimensions),
					            preprocessed.get() + (embedding_idx * num_dimensions),
					            num_dimensions * sizeof(float));
				}
			}
			StoreClusterEmbeddings<Q, embedding_storage_t>(cluster, index, tmp_cluster_embeddings.get(),
			                                               cluster_size);
		}
		searcher = std::make_unique<PDX::PDXearch<Q>>(index, *pruner);

		// L0 index
		auto l0_num_clusters = config.num_meso_clusters;
		if (l0_num_clusters == 0){
			l0_num_clusters = static_cast<uint32_t>(std::sqrt(num_clusters));
		}

		index.l0 = PDX::IndexPDXIVF<F32>(num_dimensions, num_clusters, l0_num_clusters, normalize);
		KMeansResult l0_kmeans_result = ComputeKMeans(index.centroids.data(), num_clusters, num_dimensions,
		                                           l0_num_clusters, config.distance_metric, config.seed,
		                                           config.normalize, 1.0f, 10); // No sampling for l0
		index.l0.centroids = std::move(l0_kmeans_result.centroids);

		size_t l0_max_cluster_size = 0;
		for (size_t i = 0; i < l0_num_clusters; i++) {
			l0_max_cluster_size = std::max(l0_max_cluster_size, l0_kmeans_result.assignments[i].size());
		}
		std::unique_ptr<float[]> l0_tmp_cluster_embeddings(
		    new float[static_cast<uint64_t>(l0_max_cluster_size) * num_dimensions]);

		for (size_t cluster_idx = 0; cluster_idx < l0_num_clusters; cluster_idx++) {
			const auto cluster_size = l0_kmeans_result.assignments[cluster_idx].size();
			auto &cluster = index.l0.clusters.emplace_back(cluster_size, num_dimensions);

			for (size_t position_in_cluster = 0; position_in_cluster < cluster_size; position_in_cluster++) {
				const auto embedding_idx = l0_kmeans_result.assignments[cluster_idx][position_in_cluster];
				cluster.indices[position_in_cluster] = embedding_idx;

				std::memcpy(l0_tmp_cluster_embeddings.get() + (position_in_cluster * num_dimensions),
							index.centroids.data() + (embedding_idx * num_dimensions),
							num_dimensions * sizeof(float));
			}
			StoreClusterEmbeddings<F32, float>(cluster, index.l0, l0_tmp_cluster_embeddings.get(),
			                                   cluster_size);
		}

		top_level_searcher = std::make_unique<PDX::PDXearch<F32>>(index.l0, *pruner);
	}

	void SetNProbe(uint32_t n_probe) const {
		searcher->SetNProbe(n_probe);
	}

	const PDX::PDXearch<Q> &GetSearcher() const {
		return *searcher;
	}

	uint32_t GetNumDimensions() const {
		return index.num_dimensions;
	}

	uint32_t GetNumClusters() const {
		return index.num_clusters;
	}

	uint32_t GetTopLevelNumClusters() const {
		return index.l0.num_clusters;
	}

};

using PDXIndexF32 = PDXIndex<PDX::F32>;
using PDXIndexU8 = PDXIndex<PDX::U8>;
using PDXTreeIndexF32 = PDXTreeIndex<PDX::F32>;
using PDXTreeIndexU8 = PDXTreeIndex<PDX::U8>;

} // namespace PDX
