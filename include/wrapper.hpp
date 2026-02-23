#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "common.hpp"
#include "db_mock/predicate_evaluator.hpp"
#include "index_base/pdx_ivf.hpp"
#include "pruners/adsampling.hpp"
#include "pdxearch.hpp"
#include "clustering.hpp"

namespace PDX {

class PDXearchWrapper {
public:
	static constexpr PDX::DistanceMetric DEFAULT_DISTANCE_METRIC = PDX::DistanceMetric::L2SQ;
	static constexpr PDX::Quantization DEFAULT_QUANTIZATION = PDX::Quantization::U8;
	static constexpr int32_t DEFAULT_N_PROBE = 24;

private:
	const uint32_t num_dimensions;
	// Whether the embeddings (both the query embeddings and those stored in the
	// index) are normalized. The PDXearch kernel only implements Euclidian
	// distance (L2SQ). To compute the cosine and inner product distances we
	// normalize and then use L2SQ.
	const bool is_normalized;

	// The fields below are index options that can be set during index creation
	const PDX::DistanceMetric distance_metric;
	const PDX::Quantization quantization;
	const uint32_t n_probe;
	const int32_t seed;

protected:
	const unique_ptr<float[]> rotation_matrix;

public:
	PDXearchWrapper(PDX::Quantization quantization, PDX::DistanceMetric distance_metric, uint32_t num_dimensions,
	                uint32_t n_probe, int32_t seed)
	    : num_dimensions(num_dimensions), is_normalized(DistanceMetricRequiresNormalization(distance_metric)),
	      distance_metric(distance_metric), quantization(quantization), n_probe(n_probe), seed(seed),
	      rotation_matrix(GenerateRandomRotationMatrix(num_dimensions, seed)) {
	}
	virtual ~PDXearchWrapper() = default;

	void Insert(row_t row_id, const float *embedding) {
		throw NotImplementedException("PDXearchWrapper::Insert() not implemented");
	}
	void Delete(row_t row_id) {
		throw NotImplementedException("PDXearchWrapper::Delete() not implemented");
	}

	uint64_t GetInMemorySize() const {
		throw NotImplementedException("PDXearchWrapper::GetInMemorySize() not implemented");
	}

	uint32_t GetNumDimensions() const {
		return num_dimensions;
	}
	PDX::DistanceMetric GetDistanceMetric() const {
		return distance_metric;
	}
	PDX::Quantization GetQuantization() const {
		return quantization;
	}
	bool IsNormalized() const {
		return is_normalized;
	}
	uint32_t GetNProbe() const {
		return n_probe;
	}
	int32_t GetSeed() const {
		return seed;
	}
	float *GetRotationMatrix() const {
		return rotation_matrix.get();
	}
};

struct RowIdClusterMapping {
	uint32_t cluster_id;
	uint32_t index_in_cluster;
};

// The PDXearchWrapper for the global implementation
template <PDX::Quantization Q>
class PDXearchWrapperGlobal : public PDXearchWrapper {
public:
	using embedding_storage_t = PDX::pdx_data_t<Q>;

private:
	uint32_t num_clusters {};
	uint64_t total_num_embeddings {};
	std::vector<RowIdClusterMapping> row_id_cluster_mapping;

	std::unique_ptr<PDX::IndexPDXIVF<Q>> index;
	std::unique_ptr<PDX::ADSamplingPruner<Q>> pruner;
	std::unique_ptr<PDX::PDXearch<Q>> searcher;

public:
	PDXearchWrapperGlobal(PDX::DistanceMetric distance_metric, uint32_t num_dimensions, uint32_t n_probe, int32_t seed,
	                      size_t estimated_cardinality)
	    : PDXearchWrapper(Q, distance_metric, num_dimensions, n_probe, seed),
	      num_clusters(ComputeNumberOfClusters(estimated_cardinality)), total_num_embeddings(estimated_cardinality),
	      row_id_cluster_mapping(estimated_cardinality),
	      pruner(make_uniq<PDX::ADSamplingPruner<Q>>(num_dimensions, rotation_matrix.get())) {
		// Additional constraints on the number of dimensions are enforced in `pdxearch_index_plan.cpp`.
		D_ASSERT(num_dimensions > 0);
		D_ASSERT(estimated_cardinality > 0);
		D_ASSERT(num_clusters > 0);
	}

	// Initialize the wrapper's state. This is called once.
	void SetUpIndex(const row_t *const row_ids, const float *const embeddings, const size_t num_embeddings) {
		D_ASSERT(num_embeddings == total_num_embeddings);

		const auto num_dimensions = GetNumDimensions();

		float quantization_base = 0.0f;
		float quantization_scale = 1.0f;
		if constexpr (Q == PDX::U8) {
			const auto params = PDX::ScalarQuantizer<Q>::ComputeQuantizationParams(
			    embeddings, static_cast<size_t>(num_embeddings) * num_dimensions);
			quantization_base = params.quantization_base;
			quantization_scale = params.quantization_scale;
			index = make_uniq<PDX::IndexPDXIVF<Q>>(num_dimensions, num_embeddings, num_clusters, IsNormalized(),
			                                       quantization_scale, quantization_base);
		} else {
			index = make_uniq<PDX::IndexPDXIVF<Q>>(num_dimensions, num_embeddings, num_clusters, IsNormalized());
		}

		// Compute K-means centroids and embedding-to-centroid assignment (always on float embeddings).
		KMeansResult kmeans_result =
		    ComputeKMeans(embeddings, num_embeddings, num_dimensions, num_clusters, GetDistanceMetric(), GetSeed());

		// Store centroids.
		index->centroids = std::move(kmeans_result.centroids);

		// Row-major buffer that the current cluster's embeddings are "gathered" into. This buffer is the source for
		// StoreClusterEmbeddings, the result of which is persistently stored in the index. The buffer is reused across
		// clusters. For F32: buffer is float. For U8: buffer is uint8_t (quantized).
		size_t max_cluster_size = 0;
		for (size_t i = 0; i < num_clusters; i++) {
			max_cluster_size = std::max(max_cluster_size, kmeans_result.assignments[i].size());
		}
		auto tmp_cluster_embeddings =
		    std::make_unique<embedding_storage_t[]>(static_cast<uint64_t>(max_cluster_size * num_dimensions));

		// Set up the IVF clusters' metadata and store the embeddings.
		for (size_t cluster_idx = 0; cluster_idx < num_clusters; cluster_idx++) {
			const auto cluster_size = kmeans_result.assignments[cluster_idx].size();
			auto &cluster = index->clusters.emplace_back(cluster_size, num_dimensions);

			for (size_t position_in_cluster = 0; position_in_cluster < cluster_size; position_in_cluster++) {
				const auto embedding_idx = kmeans_result.assignments[cluster_idx][position_in_cluster];
				const row_t row_id = row_ids[embedding_idx];

				D_ASSERT(row_id < total_num_embeddings);
				(row_id_cluster_mapping)[row_id] = {static_cast<uint32_t>(cluster_idx),
				                                    static_cast<uint32_t>(position_in_cluster)};
				cluster.indices[position_in_cluster] = row_id;

				if constexpr (Q == PDX::U8) {
					PDX::ScalarQuantizer<Q> quantizer(num_dimensions);
					quantizer.QuantizeEmbedding(embeddings + (embedding_idx * num_dimensions), quantization_base,
					                            quantization_scale,
					                            tmp_cluster_embeddings.get() + (position_in_cluster * num_dimensions));
				} else {
					memcpy(tmp_cluster_embeddings.get() + (position_in_cluster * num_dimensions),
					       embeddings + (embedding_idx * num_dimensions), num_dimensions * sizeof(float));
				}
			}

			StoreClusterEmbeddings<Q, embedding_storage_t>(cluster, *index, tmp_cluster_embeddings.get(), cluster_size);
		}

		// Note: the searcher depends on a fully initialized index in its constructor.
		searcher = make_uniq<PDX::PDXearch<Q>>(*index, *pruner);
	}

	std::unique_ptr<std::vector<row_t>> Search(const float *const query_embedding, const size_t limit,
	                                           const uint32_t n_probe) const {
		searcher->SetNProbe(n_probe);
		const std::vector<PDX::KNNCandidate> results = searcher->Search(query_embedding, limit);
		std::unique_ptr<std::vector<row_t>> row_ids = make_uniq<std::vector<row_t>>(results.size());
		for (size_t i = 0; i < results.size(); i++) {
			(*row_ids)[i] = results[i].index;
		}
		return row_ids;
	}

	PDX::PredicateEvaluator CreatePredicateEvaluator(std::vector<std::pair<Vector, size_t>> &row_id_vectors) const {
		auto predicate_evaluator = PDX::PredicateEvaluator(num_clusters, total_num_embeddings);

		for (auto &[row_id_vector, vector_size] : row_id_vectors) {
			row_id_vector.Flatten(vector_size);
			const auto row_id_data = FlatVector::GetData<row_t>(row_id_vector);
			const auto &validity = FlatVector::Validity(row_id_vector);

			for (size_t i = 0; i < vector_size; i++) {
				if (validity.RowIsValid(i)) {
					const auto &[cluster_id, index_in_cluster] = (row_id_cluster_mapping)[row_id_data[i]];
					predicate_evaluator.n_passing_tuples[cluster_id]++;
					predicate_evaluator.selection_vector[searcher->cluster_offsets[cluster_id] + index_in_cluster] = 1;
				}
			}
		}

		return predicate_evaluator;
	}

	std::unique_ptr<std::vector<row_t>> FilteredSearch(const float *const query_embedding, const size_t limit,
	                                                   std::vector<std::pair<Vector, size_t>> &row_id_vectors,
	                                                   const uint32_t n_probe) const {
		const PDX::PredicateEvaluator predicate_evaluator = CreatePredicateEvaluator(row_id_vectors);

		searcher->SetNProbe(n_probe);
		std::vector<PDX::KNNCandidate> results =
		    searcher->FilteredSearch(query_embedding, limit, predicate_evaluator);
		std::unique_ptr<std::vector<row_t>> row_ids = make_uniq<std::vector<row_t>>(results.size());
		for (size_t i = 0; i < results.size(); i++) {
			(*row_ids)[i] = results[i].index;
		}
		return row_ids;
	}

	size_t GetNumClusters() const {
		return num_clusters;
	}
};

using PDXearchWrapperGlobalF32 = PDXearchWrapperGlobal<PDX::F32>;
using PDXearchWrapperGlobalU8 = PDXearchWrapperGlobal<PDX::U8>;

} // namespace PDX
