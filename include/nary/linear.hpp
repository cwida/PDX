#ifndef PDX_BRUTE_FORCE_SEARCH_HPP
#define PDX_BRUTE_FORCE_SEARCH_HPP

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstring>
#include <queue>
#include "vector_searcher.hpp"
#include "pdx/index_base/pdx_ivf.hpp"
#include "utils/tictoc.hpp"
#include "pdx/distance_computers/base_computers.hpp"

/******************************************************************
 * Linear scans on entire collection and IVF buckets (only L2 w/ float32)
 * Performance on-par to FAISS
 ******************************************************************/
template<PDX::Quantization q=PDX::F32>
class LinearSearcher: public VectorSearcher<q> {

    using KNNCandidate = PDX::KNNCandidate<q>;
    using VectorComparator = PDX::VectorComparator<q>;
    using IndexPDXIVF = PDX::IndexPDXIVF<q>;
    using Vectorgroup = PDX::Vectorgroup<q>;
    using DISTANCE_TYPE = PDX::DistanceType_t<q>;
    using VECTOR_DATA_TYPE = PDX::DataType_t<q>;
    using QUERY_TYPE = PDX::QuantizedVectorType_t<q>;

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

    void PreprocessQuery(const float * raw_query, QUERY_TYPE *query){
        for (size_t i = 0; i < num_dimensions; i++){
            query[i] = static_cast<QUERY_TYPE>(raw_query[i]);
        }
    };

    std::vector<KNNCandidate> Search(const float *raw_query, const VECTOR_DATA_TYPE *data, uint32_t knn) {
#ifdef BENCHMARK_TIME
        this->ResetClocks();
        this->end_to_end_clock.Tic();
#endif

        QUERY_TYPE query[num_dimensions];
        PreprocessQuery(raw_query, query);

        best_k = std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator>{};
        for (size_t vector_idx = 0; vector_idx < num_embeddings; ++vector_idx) {
            DISTANCE_TYPE current_distance = PDX::DistanceComputer<PDX::DistanceFunction::L2, q>::Horizontal(query, data + (vector_idx * num_dimensions), num_dimensions);
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
        this->end_to_end_clock.Toc();
#endif
        return result;
    }

    std::vector<KNNCandidate> SearchIVF(const float *raw_query, uint32_t knn, IndexPDXIVF& ivf_data, size_t ivf_nprobe) {
#ifdef BENCHMARK_TIME
        this->ResetClocks();
        this->end_to_end_clock.Tic();
#endif
        this->GetVectorgroupsAccessOrderIVF(raw_query, ivf_data, ivf_nprobe, vectorgroup_indices);

        QUERY_TYPE query[num_dimensions];
        PreprocessQuery(raw_query, query);

        size_t buckets_to_visit = ivf_nprobe;
        best_k = std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator>{};
        for (size_t bucket_idx = 0; bucket_idx < buckets_to_visit; ++bucket_idx) {
            Vectorgroup &bucket = ivf_data.vectorgroups[vectorgroup_indices[bucket_idx]];
            for (size_t vector_idx = 0; vector_idx < bucket.num_embeddings; ++vector_idx) {
                DISTANCE_TYPE current_distance = PDX::DistanceComputer<PDX::DistanceFunction::L2, q>::Horizontal(bucket.data + (vector_idx * num_dimensions), query, num_dimensions);
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
        this->end_to_end_clock.Toc();
#endif
        return result;
    }
};

#endif //PDX_BRUTE_FORCE_SEARCH_HPP
