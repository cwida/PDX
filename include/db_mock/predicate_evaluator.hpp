#ifndef PDX_PREDICATE_EVALUATOR_H
#define PDX_PREDICATE_EVALUATOR_H

#include <memory>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include "common.hpp"
#include "utils/file_reader.hpp"

namespace PDX {

class PredicateEvaluator {

public:

    std::unique_ptr<char[]> file_buffer;

    uint32_t * n_passing_tuples = nullptr;
    uint8_t * selection_vector = nullptr;
    size_t passing_tuples = 0;
    size_t n_clusters;

    PredicateEvaluator(size_t n_clusters): n_clusters(n_clusters){};

    ~PredicateEvaluator() = default;

    void LoadSelectionVectorFromFile(const std::string &filename) {
        file_buffer = MmapFile(filename);
        LoadSelectionVector(file_buffer.get());
    }

    void LoadSelectionVector(char *input) {
        char *next_value = input;
        n_passing_tuples = (uint32_t *) next_value;
        passing_tuples = 0;
        for (size_t i = 0; i < n_clusters; i++) {
            passing_tuples += n_passing_tuples[i];
        }
        next_value += sizeof(uint32_t) * n_clusters;
        selection_vector = (uint8_t *) next_value;
    }

    void LoadSelectionVector(uint32_t *n_passing_tuples_p, uint8_t *selection_vector_p) {
        n_passing_tuples = n_passing_tuples_p;
        passing_tuples = 0;
        for (size_t i = 0; i < n_clusters; i++) {
            passing_tuples += n_passing_tuples[i];
        }
        selection_vector = selection_vector_p;
    }

    std::pair<uint8_t*, uint32_t> GetSelectionVector(const size_t cluster_id, const size_t cluster_offset) const {
        return { selection_vector + cluster_offset, n_passing_tuples[cluster_id]};
    }

};

}; // namespace PDX

#endif //PDX_PREDICATE_EVALUATOR_H
