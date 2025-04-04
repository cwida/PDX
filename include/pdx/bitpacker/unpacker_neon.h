#ifndef EMBEDDINGSEARCH_UNPACKER_HPP
#define EMBEDDINGSEARCH_UNPACKER_HPP


#include <queue>
#include <cassert>
#include <algorithm>
#include <array>
#include <cstdint>

namespace PDX {

class Unpacker {
public:
    /* we packed 32 6-bit values, touching 3 64-bit words, using 24 bytes */
    static void unpackblock6(const uint8_t **pw, uint8_t ** pout){
        const uint64_t * pw64 = *(const uint64_t **) pw;
        uint8_t * out = *pout;
        const uint64_t mask = UINT64_C(63);
        /* we are going to access  3 64-bit words */
        uint64_t w0 = pw64[0];
        uint64_t w1 = pw64[1];
        uint64_t w2 = pw64[2];
        *pw += 24; /* we used up 24 input bytes */
        out[0] = (uint8_t)  ( ( w0 )  & mask  );
        out[1] = (uint8_t)  ( ( w0 >> 6 )  & mask  );
        out[2] = (uint8_t)  ( ( w0 >> 12 )  & mask  );
        out[3] = (uint8_t)  ( ( w0 >> 18 )  & mask  );
        out[4] = (uint8_t)  ( ( w0 >> 24 )  & mask  );
        out[5] = (uint8_t)  ( ( w0 >> 30 )  & mask  );
        out[6] = (uint8_t)  ( ( w0 >> 36 )  & mask  );
        out[7] = (uint8_t)  ( ( w0 >> 42 )  & mask  );
        out[8] = (uint8_t)  ( ( w0 >> 48 )  & mask  );
        out[9] = (uint8_t)  ( ( w0 >> 54 )  & mask  );
        out[10] = (uint8_t)  ( ( ( w0 >> 60  ) | ( w1 << 4 ) )  & mask  );
        out[11] = (uint8_t)  ( ( w1 >> 2 )  & mask  );
        out[12] = (uint8_t)  ( ( w1 >> 8 )  & mask  );
        out[13] = (uint8_t)  ( ( w1 >> 14 )  & mask  );
        out[14] = (uint8_t)  ( ( w1 >> 20 )  & mask  );
        out[15] = (uint8_t)  ( ( w1 >> 26 )  & mask  );
        out[16] = (uint8_t)  ( ( w1 >> 32 )  & mask  );
        out[17] = (uint8_t)  ( ( w1 >> 38 )  & mask  );
        out[18] = (uint8_t)  ( ( w1 >> 44 )  & mask  );
        out[19] = (uint8_t)  ( ( w1 >> 50 )  & mask  );
        out[20] = (uint8_t)  ( ( w1 >> 56 )  & mask  );
        out[21] = (uint8_t)  ( ( ( w1 >> 62  ) | ( w2 << 2 ) )  & mask  );
        out[22] = (uint8_t)  ( ( w2 >> 4 )  & mask  );
        out[23] = (uint8_t)  ( ( w2 >> 10 )  & mask  );
        out[24] = (uint8_t)  ( ( w2 >> 16 )  & mask  );
        out[25] = (uint8_t)  ( ( w2 >> 22 )  & mask  );
        out[26] = (uint8_t)  ( ( w2 >> 28 )  & mask  );
        out[27] = (uint8_t)  ( ( w2 >> 34 )  & mask  );
        out[28] = (uint8_t)  ( ( w2 >> 40 )  & mask  );
        out[29] = (uint8_t)  ( ( w2 >> 46 )  & mask  );
        out[30] = (uint8_t)  ( ( w2 >> 52 )  & mask  );
        out[31] = (uint8_t) ( w2  >> 58  );
        *pout += 32; /* we wrote 32 32-bit integers */
    }

    static void unpackblock4(const uint8_t ** pw, uint8_t ** pout){
        const uint64_t * pw64 = *(const uint64_t **) pw;
        uint8_t * out = *pout;
        const uint64_t mask = UINT64_C(15);
        /* we are going to access  2 64-bit words */
        uint64_t w0 = pw64[0];
        uint64_t w1 = pw64[1];
        *pw += 16; /* we used up 16 input bytes */
        out[0] = (uint8_t)  ( ( w0 )  & mask  );
        out[1] = (uint8_t)  ( ( w0 >> 4 )  & mask  );
        out[2] = (uint8_t)  ( ( w0 >> 8 )  & mask  );
        out[3] = (uint8_t)  ( ( w0 >> 12 )  & mask  );
        out[4] = (uint8_t)  ( ( w0 >> 16 )  & mask  );
        out[5] = (uint8_t)  ( ( w0 >> 20 )  & mask  );
        out[6] = (uint8_t)  ( ( w0 >> 24 )  & mask  );
        out[7] = (uint8_t)  ( ( w0 >> 28 )  & mask  );
        out[8] = (uint8_t)  ( ( w0 >> 32 )  & mask  );
        out[9] = (uint8_t)  ( ( w0 >> 36 )  & mask  );
        out[10] = (uint8_t)  ( ( w0 >> 40 )  & mask  );
        out[11] = (uint8_t)  ( ( w0 >> 44 )  & mask  );
        out[12] = (uint8_t)  ( ( w0 >> 48 )  & mask  );
        out[13] = (uint8_t)  ( ( w0 >> 52 )  & mask  );
        out[14] = (uint8_t)  ( ( w0 >> 56 )  & mask  );
        out[15] = (uint8_t) ( w0  >> 60  );
        out[16] = (uint8_t)  ( ( w1 )  & mask  );
        out[17] = (uint8_t)  ( ( w1 >> 4 )  & mask  );
        out[18] = (uint8_t)  ( ( w1 >> 8 )  & mask  );
        out[19] = (uint8_t)  ( ( w1 >> 12 )  & mask  );
        out[20] = (uint8_t)  ( ( w1 >> 16 )  & mask  );
        out[21] = (uint8_t)  ( ( w1 >> 20 )  & mask  );
        out[22] = (uint8_t)  ( ( w1 >> 24 )  & mask  );
        out[23] = (uint8_t)  ( ( w1 >> 28 )  & mask  );
        out[24] = (uint8_t)  ( ( w1 >> 32 )  & mask  );
        out[25] = (uint8_t)  ( ( w1 >> 36 )  & mask  );
        out[26] = (uint8_t)  ( ( w1 >> 40 )  & mask  );
        out[27] = (uint8_t)  ( ( w1 >> 44 )  & mask  );
        out[28] = (uint8_t)  ( ( w1 >> 48 )  & mask  );
        out[29] = (uint8_t)  ( ( w1 >> 52 )  & mask  );
        out[30] = (uint8_t)  ( ( w1 >> 56 )  & mask  );
        out[31] = (uint8_t) ( w1  >> 60  );
        *pout += 32; /* we wrote 32 32-bit integers */
    }

    static void unpackblock4_notoptimal(const uint8_t ** pw, uint8_t ** pout){
        const uint64_t * pw64 = *(const uint64_t **) pw;
        uint8_t * out = *pout;
        const uint64_t mask = UINT64_C(15);
        /* we are going to access  2 64-bit words */
        uint64_t w0 = __builtin_bswap64(pw64[0]);
        uint64_t w1 = __builtin_bswap64(pw64[1]);
        *pw += 16; /* we used up 16 input bytes */

        // Total of 32 values, 128 bits.
        // So start at bit 124 and go down by 4 each time
        for (int i = 0; i < 16; ++i) {
            out[i] = (uint8_t)((w0 >> (60 - i * 4)) & mask);
        }
        for (int i = 0; i < 16; ++i) {
            out[16 + i] = (uint8_t)((w1 >> (60 - i * 4)) & mask);
        }

        *pout += 32; /* we wrote 32 32-bit integers */
    }

    static void unpackblock6_notoptimal(const uint8_t **pw, uint8_t ** pout){
        const uint64_t *pw64 = *(const uint64_t **)pw;
        uint8_t *out = *pout;
        const uint64_t mask = UINT64_C(63);  // Mask to get the lower 6 bits (0x3F)

        // Unpacking 3 64-bit words (w0, w1, w2)
        uint64_t w0 = __builtin_bswap64(pw64[0]);
        uint64_t w1 = __builtin_bswap64(pw64[1]);
        uint64_t w2 = __builtin_bswap64(pw64[2]);

        *pw += 24;  // Move the pointer forward by 24 bytes (3 * 64-bit values)

        // Unpack 6 bits at a time from the 64-bit words, starting from the most significant bits
        out[0]  = (uint8_t)((w0 >> 58) & mask);           // First 6 bits from w0
        out[1]  = (uint8_t)((w0 >> 52) & mask);           // Next 6 bits from w0
        out[2]  = (uint8_t)((w0 >> 46) & mask);           // Next 6 bits from w0
        out[3]  = (uint8_t)((w0 >> 40) & mask);           // Next 6 bits from w0
        out[4]  = (uint8_t)((w0 >> 34) & mask);           // Next 6 bits from w0
        out[5]  = (uint8_t)((w0 >> 28) & mask);           // Next 6 bits from w0
        out[6]  = (uint8_t)((w0 >> 22) & mask);           // Next 6 bits from w0
        out[7]  = (uint8_t)((w0 >> 16) & mask);           // Next 6 bits from w0
        out[8]  = (uint8_t)((w0 >> 10) & mask);           // Next 6 bits from w0
        out[9]  = (uint8_t)((w0 >> 4) & mask);            // Next 6 bits from w0
        out[10] = (uint8_t)(((w0 << 2) | (w1 >> 62)) & mask);  // Cross-boundary (w0 -> w1)
        out[11] = (uint8_t)((w1 >> 56) & mask);           // Next 6 bits from w1
        out[12] = (uint8_t)((w1 >> 50) & mask);           // Next 6 bits from w1
        out[13] = (uint8_t)((w1 >> 44) & mask);           // Next 6 bits from w1
        out[14] = (uint8_t)((w1 >> 38) & mask);           // Next 6 bits from w1
        out[15] = (uint8_t)((w1 >> 32) & mask);           // Next 6 bits from w1
        out[16] = (uint8_t)((w1 >> 26) & mask);           // Next 6 bits from w1
        out[17] = (uint8_t)((w1 >> 20) & mask);           // Next 6 bits from w1
        out[18] = (uint8_t)((w1 >> 14) & mask);           // Next 6 bits from w1
        out[19] = (uint8_t)((w1 >> 8) & mask);            // Next 6 bits from w1
        out[20] = (uint8_t)((w1 >> 2) & mask);            // Next 6 bits from w1
        out[21] = (uint8_t)(((w1 << 4) | (w2 >> 60)) & mask);  // Cross-boundary (w1 -> w2)
        out[22] = (uint8_t)((w2 >> 54) & mask);           // Next 6 bits from w2
        out[23] = (uint8_t)((w2 >> 48) & mask);           // Next 6 bits from w2
        out[24] = (uint8_t)((w2 >> 42) & mask);           // Next 6 bits from w2
        out[25] = (uint8_t)((w2 >> 36) & mask);           // Next 6 bits from w2
        out[26] = (uint8_t)((w2 >> 30) & mask);           // Next 6 bits from w2
        out[27] = (uint8_t)((w2 >> 24) & mask);           // Next 6 bits from w2
        out[28] = (uint8_t)((w2 >> 18) & mask);           // Next 6 bits from w2
        out[29] = (uint8_t)((w2 >> 12) & mask);           // Next 6 bits from w2
        out[30] = (uint8_t)((w2 >> 6) & mask);            // Next 6 bits from w2
        out[31] = (uint8_t)(w2 & mask);                   // Last 6 bits from w2

        *pout += 32;  // Move the pointer forward by 32 bytes (we wrote 32 6-bit values)

    }
};

};

#endif