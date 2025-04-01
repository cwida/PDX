#ifndef EMBEDDINGSEARCH_PXD_BOND_U8_SEARCH_HPP
#define EMBEDDINGSEARCH_PXD_BOND_U8_SEARCH_HPP

#include "pdxearch_u8.hpp"

namespace PDX {

/******************************************************************
 * BOND + PDXearch
 * No relevant functionality is added to PDXearch
 ******************************************************************/
class PDXBondSearcherU8 : public PDXearchU8<L2> {
public:
    PDXBondSearcherU8(IndexPDXIVFFlatU8 &pdx_index,
                    size_t ivf_nprobe, int position_prune_distance,
                    PDXearchDimensionsOrder dimensionOrder) :
            PDXearchU8(pdx_index,
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
