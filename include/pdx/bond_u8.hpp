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

};

} // namespace PDX

#endif //EMBEDDINGSEARCH_PXD_BOND_SEARCH_HPP
