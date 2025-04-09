#ifndef PDX_EXHAUSTIVE_BSA_SEARCH_HPP
#define PDX_EXHAUSTIVE_BSA_SEARCH_HPP

#include <cstdint>
#include <vector>
#include <queue>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstring>
#include "utils/matrix.h"
#include "vector_searcher.hpp"
#include "pdx/index_base/pdx_ivf.hpp"
#include "Eigen/Eigen/Dense"
#include "utils/tictoc.hpp"

/******************************************************************
 * Our implementation of BSA
 * https://github.com/mingyu-hkustgz/Res-Infer
 * Same as the original but improved matrix transformation performance and SIMD
 ******************************************************************/
class NaryBSASearcher: public VectorSearcher {
    Eigen::MatrixXf matrix;
    float * dimension_variances;
    float * dimension_means;
    float * base_square;
    uint32_t num_dimensions;
    uint32_t num_embeddings;
    float sigma_count;
    int delta_d;
    size_t ivf_nprobe;
    std::vector<uint32_t> vectorgroup_indices;
    std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator> best_k;

    static __attribute__((always_inline))
    inline bool UniformInference(float &tmp_dist_res, const float &threshold, int &cur_dimension, const float * pre_query){
        if (tmp_dist_res - pre_query[cur_dimension] > threshold){
            return true;
        }
        return false;
    }

    // UniformFastInference
    __attribute__((always_inline))
    inline float CalculateBSADistance(const float& threshold, const float *data_with_offset, const void *query, const float * pre_query, float result = 0, int visited_dimensions = 0) {
        // If the algorithm starts a non-zero dimensionality (visited_dimensions.e., the case of IVF++), we conduct the hypothesis testing immediately.
        if(visited_dimensions) {
            if (UniformInference(result, threshold, visited_dimensions, pre_query)){
                return -result;
            }
        }
        auto * q = (float *) query;
        auto * d = (float *) data_with_offset;

        while(visited_dimensions < (int) num_dimensions){
            // It continues to sample additional delta_d dimensions.
            int check = std::min(delta_d, ((int) num_dimensions) - visited_dimensions);
            visited_dimensions += check;
            float tmp_res = CalculateDistanceIP(d, q, check);
            d += check;
            q += check;
            result -= 2 * tmp_res;
            // Quantiles inference tesing
            if (UniformInference(result, threshold, visited_dimensions, pre_query)){
                return -result;
            }
        }
        // We return the exact distance when we have sampled all the dimensions.
        return result;
    }

public:
    NaryBSASearcher(const Eigen::MatrixXf matrix,
                    float * base_square,
                    float * dimension_variances,
                    float * dimension_means,
                    uint32_t num_dimensions,
                    uint32_t num_embeddings,
                    float sigma_count,
                    int delta_d,
                    size_t ivf_nprobe):
            matrix(matrix),
            base_square(base_square),
            dimension_variances(dimension_variances),
            dimension_means(dimension_means),
            num_dimensions(num_dimensions),
            num_embeddings(num_embeddings),
            sigma_count(sigma_count),
            delta_d(delta_d),
            ivf_nprobe(ivf_nprobe){

    };
    size_t benchmark_time = 0;

    void SetNProbe(size_t nprobe){
        ivf_nprobe = nprobe;
    }

    std::vector<KNNCandidate> SearchIVF(float *raw_query, uint32_t k, PDX::IndexPDXIVFFlat& nary_ivf_data) {
#ifdef BENCHMARK_TIME
        ResetClocks();
        end_to_end_clock.Tic();
#endif
        best_k = std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator>{};
        std::vector<float> query;
        std::vector<float> pre_query;
        float query_square;
        query.resize(num_dimensions);
        pre_query.resize(num_dimensions + 1);
        MultiplyProject(raw_query, query.data());
        GetQuerySquare(query.data(), pre_query.data(), query_square);
        GetVectorgroupsAccessOrderIVF(query.data(), nary_ivf_data, ivf_nprobe, vectorgroup_indices);
        size_t buckets_to_visit = ivf_nprobe;
        size_t points_to_visit = 0;
        for (size_t bucket_idx = 0; bucket_idx < buckets_to_visit; ++bucket_idx){
            PDX::Vectorgroup& bucket = nary_ivf_data.vectorgroups[vectorgroup_indices[bucket_idx]];
            points_to_visit += bucket.num_embeddings;
        }
        auto * distances = new float[points_to_visit];
        size_t cur_point = 0;

        // FROM 0 to DELTA_D
        float distance_limit = std::numeric_limits<float>::max();
        for (size_t bucket_idx = 0; bucket_idx < buckets_to_visit; ++bucket_idx){
            PDX::Vectorgroup& bucket = nary_ivf_data.vectorgroups[vectorgroup_indices[bucket_idx]];
            for (size_t vector_idx = 0; vector_idx < bucket.num_embeddings; ++vector_idx) {
                float distance = GetPreSum(
                        query_square,
                        bucket.indices[vector_idx]
                        ) - 2 * CalculateDistanceIP(bucket.data + (vector_idx * delta_d), query.data(), delta_d);
                distances[cur_point] = distance;
                cur_point++;
            }
        }

        cur_point = 0;
        // FROM DELTA_D to num_dimensions
        for (size_t bucket_idx = 0; bucket_idx < buckets_to_visit; ++bucket_idx){
            PDX::Vectorgroup& bucket = nary_ivf_data.vectorgroups[vectorgroup_indices[bucket_idx]];
            float * res_data = bucket.data + (bucket.num_embeddings * delta_d);
            for (size_t vector_idx = 0; vector_idx < bucket.num_embeddings; ++vector_idx) {
                // UniformFastInference
                float distance = CalculateBSADistance(distance_limit,
                                                             res_data + (vector_idx * (num_dimensions - delta_d)),
                                                             query.data() + delta_d, pre_query.data(), distances[cur_point],
                                                             delta_d);
                cur_point++;
                if (distance > 0) {
                    KNNCandidate embedding{};
                    embedding.index = bucket.indices[vector_idx];
                    embedding.distance = distance;
                    best_k.emplace(embedding);
                    if (best_k.size() > k) {
                        best_k.pop();
                    }
                }
                if (best_k.size() == k && best_k.top().distance < distance_limit) {
                    distance_limit = best_k.top().distance;
                }
            }
        }

        std::vector<KNNCandidate> result;
        result.resize(k);
        for (size_t i = 0; i < k && !best_k.empty(); ++i) {
            result[i] = best_k.top();
            best_k.pop();
        }
#ifdef BENCHMARK_TIME
        end_to_end_clock.Toc();
#endif
        delete[] distances;
        return result;
    }

    // Improved transformation (2x - 100x faster than the original one, depending on D)
    void MultiplyProject(float *raw_query, float *query) {
        std::vector<float> intermediate_query;
        intermediate_query.resize(num_dimensions);
        for (size_t i = 0; i < num_dimensions; i++){
            intermediate_query[i] = raw_query[i] - dimension_means[i];
        }
        Eigen::MatrixXf query_matrix = Eigen::Map<Eigen::MatrixXf>(intermediate_query.data(), 1, num_dimensions);
        Eigen::MatrixXf mul_result(1, num_dimensions);
        mul_result = query_matrix * matrix;
        for (size_t i = 0; i < num_dimensions; i++){
            // https://github.com/mingyu-hkustgz/Res-Infer/blob/nolearn/include/pca.h#L52
            query[i] = mul_result(0, i) - dimension_means[i];
        }
    }

    void GetQuerySquare(float *query, float *pre_query, float &query_square){
        query_square = 0;
        for (int i = 0; i < num_dimensions; i++) {
            query_square += query[i] * query[i];
            pre_query[i] = query[i] * query[i] * dimension_variances[i];
        }
        pre_query[num_dimensions] = 0;
        for (int i = (int) num_dimensions - 1; i >= 0; i--) {
            pre_query[i] += pre_query[i + 1];
        }
        for (int i = 0; i < num_dimensions; i++) {
            pre_query[i] = sqrt(pre_query[i]);
            pre_query[i] *= sigma_count * 2;
        }
    }

    inline float GetPreSum(float query_square, size_t vector_idx){
        float res = base_square[vector_idx] + query_square + (float) 1e-5;
        return res;
    }

};

#endif //PDX_EXHAUSTIVE_BSA_SEARCH_HPP
