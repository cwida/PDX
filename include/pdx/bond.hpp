#ifndef EMBEDDINGSEARCH_PXD_BOND_SEARCH_HPP
#define EMBEDDINGSEARCH_PXD_BOND_SEARCH_HPP

#include "pdxearch.hpp"

namespace PDX {

/******************************************************************
 * BOND + PDXearch
 * No relevant functionality is added to PDXearch
 ******************************************************************/
class PDXBondSearcher : public PDXearch<L2> {
public:
    PDXBondSearcher(IndexPDXIVFFlat &pdx_index, float selectivity_threshold,
                    size_t ivf_nprobe, int position_prune_distance,
                    PDXearchDimensionsOrder dimensionOrder) :
            PDXearch(pdx_index,
                     selectivity_threshold,
                     ivf_nprobe,
                     position_prune_distance,
                     dimensionOrder) {};

    // TODO: This should not be needed, however the overhead is still meaningless
    void PreprocessQuery(float *raw_query, float *query) override {
        memcpy((void *) query, (void *) raw_query, pdx_data.num_dimensions * sizeof(float));
    }

};

} // namespace PDX

#endif //EMBEDDINGSEARCH_PXD_BOND_SEARCH_HPP
