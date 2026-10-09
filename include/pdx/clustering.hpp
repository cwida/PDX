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

} // namespace PDX
