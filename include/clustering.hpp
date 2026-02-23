#pragma once

#include <vector>
#include "common.hpp"
#include "superkmeans/hierarchical_superkmeans.h"

namespace PDX {

struct KMeansResult {
	// Row-major buffer of all centroids (num_clusters * num_dimensions).
	std::vector<float> centroids;

	// Mapping from a centroid to its embeddings.
	//
	// The embeddings are represented as indices into the original `embeddings` array. The `embeddings` array was passed
	// as a parameter to the `ComputeKMeans` function.
	//
	// `assignments[0] -> [1, 3]` means that the 2nd and 4th embeddings in the `embeddings` array belong to the 0th
	// cluster/centroid.
	std::vector<std::vector<uint64_t>> assignments;

	static constexpr size_t MIN_EMBEDDINGS_TO_SAMPLE = 30720;

	explicit KMeansResult(uint32_t num_clusters) : assignments(num_clusters) {
	}
};

// Compute centroids (clusters) and centroid-to-embedding assignments using SuperKMeans.
[[nodiscard]] inline KMeansResult ComputeKMeans(const float *const embeddings, const uint64_t num_embeddings,
                                                const uint32_t num_dimensions, const uint32_t num_clusters,
                                                const PDX::DistanceMetric distance_metric, const uint32_t seed) {
	D_ASSERT(num_embeddings >= 1);
	D_ASSERT(num_dimensions >= 1);
	D_ASSERT(num_clusters >= 1);

	auto result = KMeansResult(num_clusters);

	if (num_clusters == 1) {
		// Copy the first embedding as the centroid
		result.centroids = std::vector<float>(embeddings, embeddings + num_dimensions);
		// Assign all embeddings to the first cluster
		for (uint64_t vec_id = 0; vec_id < num_embeddings; vec_id++) {
			result.assignments[0].emplace_back(vec_id);
		}
		return result;
	}

	// Compute centroids
	skmeans::HierarchicalSuperKMeansConfig config;
	if (num_embeddings < KMeansResult::MIN_EMBEDDINGS_TO_SAMPLE) {
		config.sampling_fraction = 1.0f;
	} else {
		config.sampling_fraction = 0.3f;
	}
	config.angular = distance_metric == PDX::DistanceMetric::COSINE || distance_metric == PDX::DistanceMetric::IP;
	config.data_already_rotated = true;
	config.iters_mesoclustering = 3;
	config.iters_fineclustering = 5;
	config.iters_refinement = 0;
	config.suppress_warnings = true;
	config.seed = seed;

	auto kmeans = skmeans::HierarchicalSuperKMeans(num_clusters, num_dimensions, config);
	result.centroids = kmeans.Train(embeddings, num_embeddings);

	// Extract assignments
	// SuperKMeans returns assignment from vec_id (not row_id) to centroid_idx
	std::vector<uint32_t> assignments =
	    kmeans.Assign(embeddings, result.centroids.data(), num_embeddings, num_clusters);
	// Convert into assignment from centroid_idx to vec_id (not row_id)
	result.assignments.resize(num_clusters);
	for (uint64_t vec_id = 0; vec_id < num_embeddings; vec_id++) {
		result.assignments[assignments[vec_id]].emplace_back(vec_id);
	}

	return result;
};

} // namespace PDX
