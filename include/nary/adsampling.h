#ifndef PDX_EXHAUSTIVE_ADSAMPLING_SEARCH_HPP
#define PDX_EXHAUSTIVE_ADSAMPLING_SEARCH_HPP

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
 * Our implementation of ADSampling
 * https://github.com/gaoj0017/ADSampling
 * Same as the original but improved matrix transformation performance and SIMD
 ******************************************************************/
class NaryADSamplingSearcher: public VectorSearcher {
    Eigen::MatrixXf matrix;
    uint32_t num_dimensions;
    uint32_t num_embeddings;
    float epsilon0;
    int delta_d;
    size_t ivf_nprobe;
    std::vector<uint32_t> vectorgroup_indices;
    std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator> best_k;

    // The hypothesis testing checks whether \sqrt{D/d} dis' > (1 +  epsilon0 / \sqrt{d}) * r.
    // We equivalently check whether dis' > \sqrt{d/D} * (1 +  epsilon0 / \sqrt{d}) * r.
    inline float GetRatio(const int &visited_dimensions) const {
        if(visited_dimensions == num_dimensions) {
            return 1.0;
        }
        return 1.0 * visited_dimensions / ((int) num_dimensions) * (1.0 + epsilon0 / std::sqrt(visited_dimensions)) * (1.0 + epsilon0 / std::sqrt(visited_dimensions));
    }

    // When D, epsilon_0 and delta_d can be pre-determined, it is highly suggested to define them as constexpr and provide dataset-specific functions.
    float CalculateADSamplingDistance(const float& threshold, const float *data_with_offset, const void *query, float result = 0, int visited_dimensions = 0) {
        // If the algorithm starts a non-zero dimensionality (visited_dimensions.e., the case of IVF++), we conduct the hypothesis testing immediately.
        if(visited_dimensions && result >= threshold * GetRatio(visited_dimensions)) {
            return -result * ((int) num_dimensions) / visited_dimensions;
        }
        auto * q = (float *) query;
        auto * d = (float *) data_with_offset;

        while(visited_dimensions < (int) num_dimensions){
            // It continues to sample additional delta_d dimensions.
            int check = std::min(delta_d, ((int) num_dimensions) - visited_dimensions);
            visited_dimensions += check;
            result += CalculateDistanceL2(d, q, check);
            d += check;
            q += check;
            // Hypothesis tesing
            if(result >= threshold * GetRatio(visited_dimensions)) {
                // If the null hypothesis is reject, we return the approximate distance.
                // We return -threshold' to indicate that it's a negative object.
                return -result * ((int) num_dimensions) / visited_dimensions;
            }
        }
        // We return the exact distance when we have sampled all the dimensions.
        return result;
    }

public:
    NaryADSamplingSearcher(const Eigen::MatrixXf matrix,
                           uint32_t num_dimensions,
                           uint32_t num_embeddings,
                           float epsilon0,
                           int delta_d,
                           size_t ivf_nprobe):
                                 matrix(matrix),
                                 num_dimensions(num_dimensions),
                                 num_embeddings(num_embeddings),
                                 epsilon0(epsilon0),
                                 delta_d(delta_d),
                                 ivf_nprobe(ivf_nprobe){

    };
    size_t benchmark_time = 0;

    void SetNProbe(size_t nprobe){
        ivf_nprobe = nprobe;
    }

    std::vector<KNNCandidate> Search(float *raw_query, float *data, uint32_t k) {
#ifdef BENCHMARK_TIME
        ResetClocks();
        end_to_end_clock.Tic();
#endif
        best_k = std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator>{};
        std::vector<float> query;
        query.resize(num_dimensions);
        Multiply(raw_query, query.data());

        float distance_threshold = std::numeric_limits<float>::max();
        for (size_t vector_idx = 0; vector_idx < num_embeddings; ++vector_idx) {
            float distance = CalculateADSamplingDistance(distance_threshold, data + vector_idx * num_dimensions,
                                                         query.data());
            if (distance > 0) {
                KNNCandidate embedding{};
                embedding.index = vector_idx;
                embedding.distance = distance;
                best_k.emplace(embedding);
                if (best_k.size() > k) {
                    best_k.pop();
                }
                if (best_k.size() == k && best_k.top().distance < distance_threshold) {
                    distance_threshold = best_k.top().distance;
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
        return result;
    }

    std::vector<KNNCandidate> SearchIVF(float *raw_query, uint32_t k, PDX::IndexPDXIVFFlat& nary_ivf_data) {
#ifdef BENCHMARK_TIME
        ResetClocks();
        end_to_end_clock.Tic();
#endif
        best_k = std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator>{};
        std::vector<float> query;
        query.resize(num_dimensions);
        Multiply(raw_query, query.data());
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
            PDX::Vectorgroup& partition = nary_ivf_data.vectorgroups[vectorgroup_indices[bucket_idx]];
            for (size_t vector_idx = 0; vector_idx < partition.num_embeddings; ++vector_idx) {
                float distance = CalculateDistanceL2(partition.data + (vector_idx * delta_d),
                                                     query.data(), delta_d);
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
                float distance = CalculateADSamplingDistance(distance_limit,
                                                             res_data + (vector_idx * (num_dimensions - delta_d)),
                                                             query.data() + delta_d, distances[cur_point],
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
    void Multiply(float *raw_query, float *query) {
        Eigen::MatrixXf query_matrix = Eigen::Map<Eigen::MatrixXf>(raw_query, 1, num_dimensions);
        Eigen::MatrixXf mul_result(1, num_dimensions);
        mul_result = query_matrix * matrix;
        for (size_t i = 0; i < num_dimensions; ++i){
            query[i] = mul_result(0, i);
        }
    }
};

#endif //PDX_EXHAUSTIVE_ADSAMPLING_SEARCH_HPP
