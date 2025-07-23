#ifndef PDX_PXD_BOND_SEARCH_HPP
#define PDX_PXD_BOND_SEARCH_HPP

#include <queue>

namespace PDX {

/******************************************************************
 * BOND Pruner
 ******************************************************************/
template<Quantization q=F32>
class BondPruner {
    using DISTANCES_TYPE = DistanceType_t<q>;
    using QUANTIZED_VECTOR_TYPE = QuantizedVectorType_t<q>;
    using DATA_TYPE = DataType_t<q>;
    using VECTORGROUP_TYPE = Vectorgroup<q>;
    using KNNCandidate_t = KNNCandidate<q>;
    using VectorComparator_t = VectorComparator<q>;
public:
    uint32_t num_dimensions;

    BondPruner(uint32_t num_dimensions) : num_dimensions(num_dimensions) {};

    // TODO: Do not copy
    void PreprocessQuery(float *raw_query, float *query) {
        memcpy((void *) query, (void *) raw_query, num_dimensions * sizeof(QUANTIZED_VECTOR_TYPE));
    }

    template<Quantization Q=q>
    DistanceType_t<Q> GetPruningThreshold(
        uint32_t k,
        std::priority_queue<KNNCandidate<Q>, std::vector<KNNCandidate<Q>>, VectorComparator<Q>> &heap,
        const uint32_t current_dimension_idx
    ) {
        return heap.size() == k ? heap.top().distance : std::numeric_limits<DistanceType_t<Q>>::max();
    }

};

} // namespace PDX

#endif //PDX_PXD_BOND_SEARCH_HPP
