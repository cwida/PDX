#ifndef EMBEDDINGSEARCH_BRUTE_FORCE_SEARCH_HPP
#define EMBEDDINGSEARCH_BRUTE_FORCE_SEARCH_HPP

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstring>
#include <queue>
#include "vector_searcher_u8.hpp"
#include "vector_searcher.hpp"
#include "pdx/index_base/pdx_ivf.hpp"
#include "utils/tictoc.hpp"

/******************************************************************
 * Linear scans on entire collection and IVF buckets (only L2 w/ uint8)
 * Performance on-par to FAISS
 ******************************************************************/
class LinearSearcherU8: public VectorSearcherU8 {
    uint32_t num_dimensions;
    uint32_t num_embeddings;
    std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator> best_k;
    std::vector<uint32_t> vectorgroup_indices;

public:
    size_t benchmark_time = 0;
    LinearSearcherU8(
            uint32_t num_dimensions,
            uint32_t num_embeddings):
            num_dimensions(num_dimensions),
            num_embeddings(num_embeddings) {

    }

    void PreprocessQuery(float * raw_query, uint8_t *query){
        for (size_t i = 0; i < num_dimensions; i++){
            query[i] = static_cast<uint8_t>(raw_query[i]);
        }
    };

    std::vector<KNNCandidate> Search(float *raw_query, const uint8_t *data, uint32_t knn) {
#ifdef BENCHMARK_TIME
        ResetClocks();
        end_to_end_clock.Tic();
#endif
        uint8_t query[num_dimensions];
        PreprocessQuery(raw_query, query);
        best_k = std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator>{};
        for (size_t vector_idx = 0; vector_idx < num_embeddings; ++vector_idx) {
            uint32_t current_distance = CalculateDistanceL2(data + (vector_idx * num_dimensions), query, num_dimensions);
            if (best_k.size() < knn || best_k.top().distance > current_distance) {
                KNNCandidate e{};
                e.index = vector_idx;
                e.distance = (float)current_distance;
                if (best_k.size() == knn) {
                    best_k.pop();
                }
                best_k.emplace(e);
            }
        }

        std::vector<KNNCandidate> result;
        result.resize(knn);
        for (size_t i = 0; i < knn && !best_k.empty(); ++i) {
            result[knn - i - 1] = best_k.top();
            best_k.pop();
        }
#ifdef BENCHMARK_TIME
        end_to_end_clock.Toc();
#endif
        return result;
    }

    /*
    std::vector<KNNCandidate> SearchIVF(float *raw_query, uint32_t knn, PDX::IndexPDXIVFFlat& ivf_data, size_t ivf_nprobe) {
#ifdef BENCHMARK_TIME
        ResetClocks();
        end_to_end_clock.Tic();
#endif
        uint8_t query[num_dimensions];
        PreprocessQuery(raw_query, query);
        GetVectorgroupsAccessOrderIVF(query, ivf_data, ivf_nprobe, vectorgroup_indices);
        size_t buckets_to_visit = ivf_nprobe;
        best_k = std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator>{};
        for (size_t bucket_idx = 0; bucket_idx < buckets_to_visit; ++bucket_idx) {
            PDX::Vectorgroup &bucket = ivf_data.vectorgroups[vectorgroup_indices[bucket_idx]];
            for (size_t vector_idx = 0; vector_idx < bucket.num_embeddings; ++vector_idx) {
                uint32_t current_distance = CalculateDistanceL2(bucket.data + (vector_idx * num_dimensions), query, num_dimensions);
                if (best_k.size() < knn || best_k.top().distance > current_distance) {
                    KNNCandidate e{};
                    e.index = bucket.indices[vector_idx];;
                    e.distance = current_distance;
                    if (best_k.size() == knn) {
                        best_k.pop();
                    }
                    best_k.emplace(e);
                }
            }
        }
        std::vector<KNNCandidate> result;
        result.resize(knn);
        for (size_t i = 0; i < knn && !best_k.empty(); ++i) {
            result[knn - i - 1] = best_k.top();
            best_k.pop();
        }
#ifdef BENCHMARK_TIME
        end_to_end_clock.Toc();
#endif
        return result;
    }
     */
};

#endif //EMBEDDINGSEARCH_BRUTE_FORCE_SEARCH_HPP
