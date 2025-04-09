#ifndef PDX_PXD_BOND_SEARCH_HPP
#define PDX_PXD_BOND_SEARCH_HPP

#include "pdxearch.hpp"

namespace PDX {

/******************************************************************
 * BOND + PDXearch
 * No relevant functionality is added to PDXearch
 ******************************************************************/
template<Quantization q=F32>
class PDXBondSearcher : public PDXearch<q> {
    using DISTANCES_TYPE = DistanceType_t<q>;
    using QUANTIZED_VECTOR_TYPE = QuantizedVectorType_t<q>;
    using DATA_TYPE = DataType_t<q>;
    using INDEX_TYPE = IndexPDXIVF<q>;
    using VECTORGROUP_TYPE = Vectorgroup<q>;
    using KNNCandidate_t = KNNCandidate<q>;
    using VectorComparator_t = VectorComparator<q>;
public:
    PDXBondSearcher(INDEX_TYPE &pdx_index,
                    size_t ivf_nprobe, int position_prune_distance,
                    DimensionsOrder dimensionOrder) :
            PDXearch<q>(pdx_index,
                     ivf_nprobe,
                     position_prune_distance,
                     dimensionOrder) {};

    // TODO: Insted of copying, do a pointer reassigning
    void PreprocessQuery(float *raw_query, float *query) override {
        memcpy((void *) query, (void *) raw_query, this->pdx_data.num_dimensions * sizeof(QUANTIZED_VECTOR_TYPE));
    }

};

} // namespace PDX

#endif //PDX_PXD_BOND_SEARCH_HPP
