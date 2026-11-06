#pragma once

#include <cstdint>
#include <cstdio>
#include <memory>

#include "superkmeans/allocator.h"

namespace PDX {

class PredicateEvaluator {

  public:
    // Number of tuples per cluster that passed the predicate.
    skmeans::AllocatedBuffer<uint32_t> n_passing_tuples;
    // Contiguous array of selection vectors for each cluster.
    // [(cluster 1): 0, 1, 1, (cluster 2): 1, 1, 1, 1].
    skmeans::AllocatedBuffer<uint8_t> selection_vector;
    size_t n_clusters;
    size_t total_passing_tuples = 0;

    PredicateEvaluator(
        size_t n_clusters,
        size_t total_num_embeddings,
        skmeans::MemoryAllocator& allocator
    )
        : n_clusters(n_clusters) {
        n_passing_tuples = skmeans::AllocateZeroedBuffer<uint32_t>(allocator, n_clusters);
        selection_vector = skmeans::AllocateZeroedBuffer<uint8_t>(allocator, total_num_embeddings);
    };

    std::pair<uint8_t*, uint32_t> GetSelectionVector(
        const size_t cluster_id,
        const size_t cluster_offset
    ) const {
        return {&selection_vector[cluster_offset], n_passing_tuples[cluster_id]};
    }
};

}; // namespace PDX
