#ifndef EMBEDDINGSEARCH_BRUTE_FORCE_SEARCH_HPP
#define EMBEDDINGSEARCH_BRUTE_FORCE_SEARCH_HPP

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstring>
#include <queue>
#include "vector_searcher.hpp"
#include "pdx/index_base/pdx_ivf.hpp"
#include "utils/tictoc.hpp"

/******************************************************************
 * Linear scans on entire collection and IVF buckets (only L2 w/ float32)
 * Performance on-par to FAISS
 ******************************************************************/
class LinearSearcher: public VectorSearcher {
    uint32_t num_dimensions;
    uint32_t num_embeddings;
    std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator> best_k;
    std::vector<uint32_t> vectorgroup_indices;

public:
    size_t benchmark_time = 0;
    LinearSearcher(
            uint32_t num_dimensions,
            uint32_t num_embeddings):
            num_dimensions(num_dimensions),
            num_embeddings(num_embeddings) {

    }

    std::vector<KNNCandidate> Search(const float *query, const float *data, uint32_t knn) {
#ifdef BENCHMARK_TIME
        ResetClocks();
        end_to_end_clock.Tic();
#endif
        best_k = std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator>{};
        for (size_t vector_idx = 0; vector_idx < num_embeddings; ++vector_idx) {
            float current_distance = CalculateDistanceL2(data + (vector_idx * num_dimensions), query, num_dimensions);
            if (best_k.size() < knn || best_k.top().distance > current_distance) {
                KNNCandidate e{};
                e.index = vector_idx;
                e.distance = current_distance;
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

    std::vector<KNNCandidate> SearchIVF(const float *query, uint32_t knn, PDX::IndexPDXIVFFlat& ivf_data, size_t ivf_nprobe) {
#ifdef BENCHMARK_TIME
        ResetClocks();
        end_to_end_clock.Tic();
#endif
#ifdef BENCHMARK_PHASES
        find_nearest_buckets_clock.Tic();
#endif
        GetVectorgroupsAccessOrderIVF(query, ivf_data, ivf_nprobe, vectorgroup_indices);
#ifdef BENCHMARK_PHASES
        find_nearest_buckets_clock.Toc();
#endif
        size_t buckets_to_visit = ivf_nprobe;
        best_k = std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator>{};
#ifdef BENCHMARK_PHASES
        distance_calculation.Tic();
#endif
        for (size_t bucket_idx = 0; bucket_idx < buckets_to_visit; ++bucket_idx) {
            PDX::Vectorgroup &bucket = ivf_data.vectorgroups[vectorgroup_indices[bucket_idx]];
            for (size_t vector_idx = 0; vector_idx < bucket.num_embeddings; ++vector_idx) {
                float current_distance = CalculateDistanceL2(bucket.data + (vector_idx * num_dimensions), query, num_dimensions);
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
#ifdef BENCHMARK_PHASES
        distance_calculation.Toc();
#endif
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
};

#endif //EMBEDDINGSEARCH_BRUTE_FORCE_SEARCH_HPP
