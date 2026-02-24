#pragma once

#include <memory>
#include <cstdint>
#include <cstdio>

namespace PDX {

class PredicateEvaluator {

public:
	// Number of tuples per cluster that passed the predicate.
	std::unique_ptr<uint32_t[]> n_passing_tuples;
	// Contiguous array of selection vectors for each cluster.
	// [(cluster 1): 0, 1, 1, (cluster 2): 1, 1, 1, 1].
	std::unique_ptr<uint8_t[]> selection_vector;
	size_t n_clusters;

	explicit PredicateEvaluator(size_t n_clusters, size_t total_num_embeddings) : n_clusters(n_clusters) {
		n_passing_tuples = std::make_unique<uint32_t[]>(n_clusters);
		selection_vector = std::make_unique<uint8_t[]>(total_num_embeddings);
	};

	std::pair<uint8_t *, uint32_t> GetSelectionVector(const size_t cluster_id, const size_t cluster_offset) const {
		return {&selection_vector[cluster_offset], n_passing_tuples[cluster_id]};
	}
};

}; // namespace PDX
