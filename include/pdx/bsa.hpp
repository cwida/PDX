#ifndef EMBEDDINGSEARCH_BSA_HPP
#define EMBEDDINGSEARCH_BSA_HPP

#include "pdxearch.hpp"
#include <Eigen/Eigen/Dense>
#include <utility>

namespace PDX {

/******************************************************************
 * BSA + PDXearch
 * Overrides many behaviours from PDXearch as BSA_pca distance calculation
 * and bounds evaluation is not quite straightforward
 ******************************************************************/
class BSASearcher : public PDXearch<IP> {
public:
    BSASearcher(IndexPDXIVFFlat &pdx_index, float selectivity_threshold,
                size_t ivf_nprobe, float sigma_count, Eigen::MatrixXf matrix,
                float *dimension_variances, float *dimension_means, float *base_square,
                PDXearchDimensionsOrder dimension_order)
            : PDXearch<IP>(pdx_index,
                           selectivity_threshold, ivf_nprobe,
                           0,
                           dimension_order),
              base_square(base_square),
              dimension_variances(dimension_variances),
              dimension_means(dimension_means),
              sigma_count(sigma_count), matrix(std::move(matrix)) {
        pre_query.resize(pdx_data.num_dimensions + 1);
    }

    inline void EvaluatePruningPredicateVectorized(uint32_t &n_pruned) override {
        for (size_t vector_idx = 0; vector_idx < PDX_VECTOR_SIZE; ++vector_idx) {
            n_pruned += (pruning_distances[vector_idx] - pre_query[current_dimension_idx]) >= pruning_threshold;
        }
    };

    inline void EvaluatePruningPredicateScalar(uint32_t &n_pruned, size_t n_vectors) override {
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            n_pruned += (pruning_distances[vector_idx] - pre_query[current_dimension_idx]) >= pruning_threshold;
        }
    };

    inline void EvaluatePruningPredicateOnPositionsArray(size_t n_vectors) override {
        n_vectors_not_pruned = 0;
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            pruning_positions[n_vectors_not_pruned] = pruning_positions[vector_idx];
            n_vectors_not_pruned +=
                    (pruning_distances[pruning_positions[vector_idx]] - pre_query[current_dimension_idx]) <
                    pruning_threshold;
        }
    };

    inline void ResetPruningDistances(size_t n_vectors) override {
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            pruning_distances[vector_idx] = GetPreSum(
                    pdx_data.vectorgroups[current_vectorgroup].indices[vector_idx]
            );
        }
    }

    inline void ResetDistancesVectorized() override {
        for (size_t vector_idx = 0; vector_idx < PDX_VECTOR_SIZE; ++vector_idx) {
            distances[vector_idx] = GetPreSum(
                    pdx_data.vectorgroups[current_vectorgroup].indices[vector_idx]
            );
        }
    }

    void InitPositionsArray(size_t n_vectors) override {
        n_vectors_not_pruned = 0;
        for (size_t vector_idx = 0; vector_idx < n_vectors; vector_idx++) {
            pruning_positions[n_vectors_not_pruned] = vector_idx;
            n_vectors_not_pruned +=
                    (pruning_distances[vector_idx] - pre_query[current_dimension_idx]) < pruning_threshold;
        }
    };

    void PreprocessQuery(float *raw_query, float *query) override {
        MultiplyProject(raw_query, query);
        GetQuerySquare(query);
    }

private:
    float sigma_count;
    Eigen::MatrixXf matrix;
    float *dimension_variances;
    float *dimension_means;
    float *base_square;
    float query_square = 0;
    std::vector<float> pre_query;

    // Improved transformation (2x - 100x faster than the original one, depending on D)
    void MultiplyProject(float *raw_query, float *query) {
        std::vector<float> intermediate_query;
        intermediate_query.resize(pdx_data.num_dimensions);
        for (size_t i = 0; i < pdx_data.num_dimensions; ++i) {
            intermediate_query[i] = raw_query[i] - dimension_means[i];
        }
        Eigen::MatrixXf query_matrix = Eigen::Map<Eigen::MatrixXf>(intermediate_query.data(), 1,
                                                                   pdx_data.num_dimensions);
        Eigen::MatrixXf mul_result(1, pdx_data.num_dimensions);
        mul_result = query_matrix * matrix;
        for (size_t i = 0; i < pdx_data.num_dimensions; ++i) {
            // https://github.com/mingyu-hkustgz/Res-Infer/blob/nolearn/include/pca.h#L52
            query[i] = mul_result(0, i) - dimension_means[i];
        }
    }

    void GetQuerySquare(float *query) {
        query_square = 0;
        for (int i = 0; i < pdx_data.num_dimensions; i++) {
            query_square += query[i] * query[i];
            pre_query[i] = query[i] * query[i] * dimension_variances[i];
        }
        pre_query[pdx_data.num_dimensions] = 0;
        for (int i = (int) pdx_data.num_dimensions - 1; i >= 0; --i) {
            pre_query[i] += pre_query[i + 1];
        }
        for (int i = 0; i < pdx_data.num_dimensions; ++i) {
            pre_query[i] = sqrt(pre_query[i]);
            pre_query[i] *= sigma_count * 2;
        }
    }

    inline float GetPreSum(size_t vector_idx) {
        float res = base_square[vector_idx] + query_square + (float) 1e-5;
        return res;
    }

};

} // namespace PDX

#endif //EMBEDDINGSEARCH_BSA_HPP
