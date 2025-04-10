#ifndef PDX_VECTOR_SEARCHER_HPP
#define PDX_VECTOR_SEARCHER_HPP

#include <vector>
#include "pdx/common.hpp"
#include "pdx/index_base/pdx_ivf.hpp"
#include "pdx/distance_computers/base_computers.hpp"
#include "utils/tictoc.hpp"


/******************************************************************
 * Base vector searcher class.
 * Contains common functionalities to any vector searcher
 ******************************************************************/
template<PDX::Quantization q=PDX::F32>
class VectorSearcher {
public:
    std::priority_queue<PDX::KNNCandidate<q>, std::vector<PDX::KNNCandidate<q>>, PDX::VectorComparator<q>> best_k{};
    TicToc end_to_end_clock = TicToc();

    void ResetClocks(){
        end_to_end_clock.Reset();
    }

    static void GetVectorgroupsAccessOrderIVF(const float *__restrict query, const PDX::IndexPDXIVF<q> &data, size_t ivf_nprobe, std::vector<uint32_t> &vectorgroups_indices) {
        std::vector<float> distances_to_centroids;
        distances_to_centroids.resize(data.num_vectorgroups);
        for (size_t vectorgroup_idx = 0; vectorgroup_idx < data.num_vectorgroups; vectorgroup_idx++) {
            distances_to_centroids[vectorgroup_idx] =
                    PDX::DistanceComputer<PDX::DistanceFunction::L2, PDX::Quantization::F32>::Horizontal(query,
                                                                          data.centroids +
                                                                          vectorgroup_idx *
                                                                          data.num_dimensions,
                                                                          data.num_dimensions);
        }
        vectorgroups_indices.resize(data.num_vectorgroups);
        std::iota(vectorgroups_indices.begin(), vectorgroups_indices.end(), 0);
        std::partial_sort(vectorgroups_indices.begin(), vectorgroups_indices.begin() + ivf_nprobe, vectorgroups_indices.end(),
                          [&distances_to_centroids](size_t i1, size_t i2) {
                              return distances_to_centroids[i1] < distances_to_centroids[i2];
                          }
        );
    }

};

#endif //PDX_VECTOR_SEARCHER_HPP
