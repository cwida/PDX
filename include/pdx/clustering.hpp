#pragma once

#include "pdx/common.hpp"
#include "superkmeans/hierarchical_superkmeans.h"
#include <vector>

namespace PDX {

struct KMeansResult {
    // Row-major buffer of all centroids (num_clusters * num_dimensions).
    std::vector<float> centroids;

    // Mapping from a centroid to its embeddings.
    //
    // The embeddings are represented as indices into the original `embeddings` array. The
    // `embeddings` array was passed as a parameter to the `ComputeKMeans` function.
    //
    // `assignments[0] -> [1, 3]` means that the 2nd and 4th embeddings in the `embeddings` array
    // belong to the 0th cluster/centroid.
    std::vector<std::vector<uint64_t>> assignments;

    static constexpr size_t MIN_EMBEDDINGS_TO_SAMPLE = 30720;

    explicit KMeansResult(uint32_t num_clusters) : assignments(num_clusters) {}
};

// Compute centroids (clusters) and centroid-to-embedding assignments using SuperKMeans.
[[nodiscard]] inline KMeansResult ComputeKMeans(
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
    const uint32_t n_threads = 0
) {
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
    if (use_hierarchical_indexing) {
        skmeans::HierarchicalSuperKMeansConfig config;
        config.sampling_fraction = 1.0f; // For now we are using all points
        config.angular = is_angular;
        config.data_already_rotated = true;
        config.suppress_warnings = true;
        config.iters_mesoclustering = 3;
        config.iters_fineclustering = 5;
        config.iters_refinement = 0;
        config.seed = seed;
        // config.verbose = true;
        config.n_threads = n_threads > 0 ? n_threads : PDX::g_n_threads;
        auto kmeans = skmeans::HierarchicalSuperKMeans(num_clusters, num_dimensions, config);
        result.centroids = kmeans.Train(embeddings, num_embeddings);
        if (num_clusters > skmeans::N_CLUSTERS_THRESHOLD_FOR_PRUNING) {
            assignments = kmeans.FastAssign(
                embeddings, result.centroids.data(), num_embeddings, num_clusters
            );
        } else {
            assignments =
                kmeans.Assign(embeddings, result.centroids.data(), num_embeddings, num_clusters);
        }
    } else {
        skmeans::SuperKMeansConfig config;
        config.sampling_fraction = chosen_sampling_fraction;
        config.angular = is_angular;
        config.data_already_rotated = true;
        config.suppress_warnings = true;
        config.iters = kmeans_iters;
        config.seed = seed;
        // config.verbose = true;
        config.n_threads = n_threads > 0 ? n_threads : PDX::g_n_threads;
        auto kmeans = skmeans::SuperKMeans(num_clusters, num_dimensions, config);
        result.centroids = kmeans.Train(embeddings, num_embeddings);
        if (num_clusters > skmeans::N_CLUSTERS_THRESHOLD_FOR_PRUNING) {
            assignments = kmeans.FastAssign(
                embeddings, result.centroids.data(), num_embeddings, num_clusters
            );
        } else {
            assignments =
                kmeans.Assign(embeddings, result.centroids.data(), num_embeddings, num_clusters);
        }
    }

    // Convert from vec_id -> centroid_idx into centroid_idx -> vec_id
    result.assignments.resize(num_clusters);
    for (uint64_t vec_id = 0; vec_id < num_embeddings; vec_id++) {
        result.assignments[assignments[vec_id]].emplace_back(vec_id);
    }

    return result;
};

} // namespace PDX
