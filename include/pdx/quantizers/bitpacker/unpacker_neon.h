#ifndef PDX_UNPACKER_HPP
#define PDX_UNPACKER_HPP


#include <queue>
#include <cassert>
#include <algorithm>
#include <array>
#include <cstdint>

#ifdef __ARM_NEON
#include "arm_neon.h"
#endif

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

    static void unpack_6bw_8ow_128crw_8uf(const uint8_t *__restrict a_in_p, uint8_t *__restrict a_out_p)
    {
        [[maybe_unused]] auto out = (a_out_p);
        [[maybe_unused]] const auto in = (a_in_p);
        [[maybe_unused]] uint8x16_t register_0;
        [[maybe_unused]] uint8x16_t tmp_0;
        [[maybe_unused]] uint8x16_t register_1;
        [[maybe_unused]] uint8x16_t tmp_1;
        [[maybe_unused]] uint8x16_t register_2;
        [[maybe_unused]] uint8x16_t tmp_2;
        [[maybe_unused]] uint8x16_t register_3;
        [[maybe_unused]] uint8x16_t tmp_3;
        [[maybe_unused]] uint8x16_t register_4;
        [[maybe_unused]] uint8x16_t tmp_4;
        [[maybe_unused]] uint8x16_t register_5;
        [[maybe_unused]] uint8x16_t tmp_5;
        [[maybe_unused]] uint8x16_t register_6;
        [[maybe_unused]] uint8x16_t tmp_6;
        [[maybe_unused]] uint8x16_t register_7;
        [[maybe_unused]] uint8x16_t tmp_7;
        [[maybe_unused]] int8x16_t base_0 = vmovq_n_u8(0ULL);
        [[maybe_unused]] int8x16_t base_1 = vmovq_n_u8(0ULL);
        [[maybe_unused]] int8x16_t base_2 = vmovq_n_u8(0ULL);
        [[maybe_unused]] int8x16_t base_3 = vmovq_n_u8(0ULL);
        [[maybe_unused]] int8x16_t base_4 = vmovq_n_u8(0ULL);
        [[maybe_unused]] int8x16_t base_5 = vmovq_n_u8(0ULL);
        [[maybe_unused]] int8x16_t base_6 = vmovq_n_u8(0ULL);
        [[maybe_unused]] int8x16_t base_7 = vmovq_n_u8(0ULL);
        uint8x16_t mask = vdupq_n_u8((1ULL << 6) - 1);

        // Decompress 768 bytes (1024 tuples of 6 bits) into 1024 bytes (1024 tuples of 8 bits)
        register_0 = vld1q_u8(in + (0 * 1 * 16) + 0);
        register_1 = vld1q_u8(in + (1 * 1 * 16) + 0);
        register_2 = vld1q_u8(in + (2 * 1 * 16) + 0);
        register_3 = vld1q_u8(in + (3 * 1 * 16) + 0);
        register_4 = vld1q_u8(in + (4 * 1 * 16) + 0);
        register_5 = vld1q_u8(in + (5 * 1 * 16) + 0);
        register_6 = vld1q_u8(in + (6 * 1 * 16) + 0);
        register_7 = vld1q_u8(in + (7 * 1 * 16) + 0);
        tmp_0 = vandq_u8(register_0, mask);
        tmp_1 = vandq_u8(register_1, mask);
        tmp_2 = vandq_u8(register_2, mask);
        tmp_3 = vandq_u8(register_3, mask);
        tmp_4 = vandq_u8(register_4, mask);
        tmp_5 = vandq_u8(register_5, mask);
        tmp_6 = vandq_u8(register_6, mask);
        tmp_7 = vandq_u8(register_7, mask);
        vst1q_u8(out + (0 * 1 * 16) + (128 * 0), tmp_0);
        vst1q_u8(out + (1 * 1 * 16) + (128 * 0), tmp_1);
        vst1q_u8(out + (2 * 1 * 16) + (128 * 0), tmp_2);
        vst1q_u8(out + (3 * 1 * 16) + (128 * 0), tmp_3);
        vst1q_u8(out + (4 * 1 * 16) + (128 * 0), tmp_4);
        vst1q_u8(out + (5 * 1 * 16) + (128 * 0), tmp_5);
        vst1q_u8(out + (6 * 1 * 16) + (128 * 0), tmp_6);
        vst1q_u8(out + (7 * 1 * 16) + (128 * 0), tmp_7);
        tmp_0 = vandq_u8(vshrq_n_u8(register_0, 6), vdupq_n_u8((1ULL << 2) - 1));
        tmp_1 = vandq_u8(vshrq_n_u8(register_1, 6), vdupq_n_u8((1ULL << 2) - 1));
        tmp_2 = vandq_u8(vshrq_n_u8(register_2, 6), vdupq_n_u8((1ULL << 2) - 1));
        tmp_3 = vandq_u8(vshrq_n_u8(register_3, 6), vdupq_n_u8((1ULL << 2) - 1));
        tmp_4 = vandq_u8(vshrq_n_u8(register_4, 6), vdupq_n_u8((1ULL << 2) - 1));
        tmp_5 = vandq_u8(vshrq_n_u8(register_5, 6), vdupq_n_u8((1ULL << 2) - 1));
        tmp_6 = vandq_u8(vshrq_n_u8(register_6, 6), vdupq_n_u8((1ULL << 2) - 1));
        tmp_7 = vandq_u8(vshrq_n_u8(register_7, 6), vdupq_n_u8((1ULL << 2) - 1));
        register_0 = vld1q_u8(in + (0 * 1 * 16) + 128);
        register_1 = vld1q_u8(in + (1 * 1 * 16) + 128);
        register_2 = vld1q_u8(in + (2 * 1 * 16) + 128);
        register_3 = vld1q_u8(in + (3 * 1 * 16) + 128);
        register_4 = vld1q_u8(in + (4 * 1 * 16) + 128);
        register_5 = vld1q_u8(in + (5 * 1 * 16) + 128);
        register_6 = vld1q_u8(in + (6 * 1 * 16) + 128);
        register_7 = vld1q_u8(in + (7 * 1 * 16) + 128);
        tmp_0 = vorrq_u8(vshlq_n_u64(vandq_u8(register_0, vdupq_n_u8((1ULL << 4) - 1)) ,2), tmp_0);
        tmp_1 = vorrq_u8(vshlq_n_u64(vandq_u8(register_1, vdupq_n_u8((1ULL << 4) - 1)) ,2), tmp_1);
        tmp_2 = vorrq_u8(vshlq_n_u64(vandq_u8(register_2, vdupq_n_u8((1ULL << 4) - 1)) ,2), tmp_2);
        tmp_3 = vorrq_u8(vshlq_n_u64(vandq_u8(register_3, vdupq_n_u8((1ULL << 4) - 1)) ,2), tmp_3);
        tmp_4 = vorrq_u8(vshlq_n_u64(vandq_u8(register_4, vdupq_n_u8((1ULL << 4) - 1)) ,2), tmp_4);
        tmp_5 = vorrq_u8(vshlq_n_u64(vandq_u8(register_5, vdupq_n_u8((1ULL << 4) - 1)) ,2), tmp_5);
        tmp_6 = vorrq_u8(vshlq_n_u64(vandq_u8(register_6, vdupq_n_u8((1ULL << 4) - 1)) ,2), tmp_6);
        tmp_7 = vorrq_u8(vshlq_n_u64(vandq_u8(register_7, vdupq_n_u8((1ULL << 4) - 1)) ,2), tmp_7);
        vst1q_u8(out +  (0 * 1 * 16) + (128 * 1), tmp_0);
        vst1q_u8(out +  (1 * 1 * 16) + (128 * 1), tmp_1);
        vst1q_u8(out + (2 * 1 * 16) + (128 * 1), tmp_2);
        vst1q_u8(out + (3 * 1 * 16) + (128 * 1), tmp_3);
        vst1q_u8(out +  (4 * 1 * 16) + (128 * 1), tmp_4);
        vst1q_u8(out +  (5 * 1 * 16) + (128 * 1), tmp_5);
        vst1q_u8(out +  (6 * 1 * 16) + (128 * 1), tmp_6);
        vst1q_u8(out +  (7 * 1 * 16) + (128 * 1), tmp_7);
        tmp_0 = vandq_u8(vshrq_n_u8(register_0, 4), vdupq_n_u8((1ULL << 4) - 1));
        tmp_1 = vandq_u8(vshrq_n_u8(register_1, 4), vdupq_n_u8((1ULL << 4) - 1));
        tmp_2 = vandq_u8(vshrq_n_u8(register_2, 4), vdupq_n_u8((1ULL << 4) - 1));
        tmp_3 = vandq_u8(vshrq_n_u8(register_3, 4), vdupq_n_u8((1ULL << 4) - 1));
        tmp_4 = vandq_u8(vshrq_n_u8(register_4, 4), vdupq_n_u8((1ULL << 4) - 1));
        tmp_5 = vandq_u8(vshrq_n_u8(register_5, 4), vdupq_n_u8((1ULL << 4) - 1));
        tmp_6 = vandq_u8(vshrq_n_u8(register_6, 4), vdupq_n_u8((1ULL << 4) - 1));
        tmp_7 = vandq_u8(vshrq_n_u8(register_7, 4), vdupq_n_u8((1ULL << 4) - 1));
        register_0 = vld1q_u8(in + (0 * 1 * 16) + 256);
        register_1 = vld1q_u8(in + (1 * 1 * 16) + 256);
        register_2 = vld1q_u8(in + (2 * 1 * 16) + 256);
        register_3 = vld1q_u8(in + (3 * 1 * 16) + 256);
        register_4 = vld1q_u8(in + (4 * 1 * 16) + 256);
        register_5 = vld1q_u8(in + (5 * 1 * 16) + 256);
        register_6 = vld1q_u8(in + (6 * 1 * 16) + 256);
        register_7 = vld1q_u8(in + (7 * 1 * 16) + 256);
        tmp_0 = vorrq_u8(vshlq_n_u64(vandq_u8(register_0, vdupq_n_u8((1ULL << 2) - 1)) ,4), tmp_0);
        tmp_1 = vorrq_u8(vshlq_n_u64(vandq_u8(register_1, vdupq_n_u8((1ULL << 2) - 1)) ,4), tmp_1);
        tmp_2 = vorrq_u8(vshlq_n_u64(vandq_u8(register_2, vdupq_n_u8((1ULL << 2) - 1)) ,4), tmp_2);
        tmp_3 = vorrq_u8(vshlq_n_u64(vandq_u8(register_3, vdupq_n_u8((1ULL << 2) - 1)) ,4), tmp_3);
        tmp_4 = vorrq_u8(vshlq_n_u64(vandq_u8(register_4, vdupq_n_u8((1ULL << 2) - 1)) ,4), tmp_4);
        tmp_5 = vorrq_u8(vshlq_n_u64(vandq_u8(register_5, vdupq_n_u8((1ULL << 2) - 1)) ,4), tmp_5);
        tmp_6 = vorrq_u8(vshlq_n_u64(vandq_u8(register_6, vdupq_n_u8((1ULL << 2) - 1)) ,4), tmp_6);
        tmp_7 = vorrq_u8(vshlq_n_u64(vandq_u8(register_7, vdupq_n_u8((1ULL << 2) - 1)) ,4), tmp_7);
        vst1q_u8(out + (0 * 1 * 16) + (128 * 2), tmp_0);
        vst1q_u8(out + (1 * 1 * 16) + (128 * 2), tmp_1);
        vst1q_u8(out + (2 * 1 * 16) + (128 * 2), tmp_2);
        vst1q_u8(out + (3 * 1 * 16) + (128 * 2), tmp_3);
        vst1q_u8(out + (4 * 1 * 16) + (128 * 2), tmp_4);
        vst1q_u8(out + (5 * 1 * 16) + (128 * 2), tmp_5);
        vst1q_u8(out + (6 * 1 * 16) + (128 * 2), tmp_6);
        vst1q_u8(out + (7 * 1 * 16) + (128 * 2), tmp_7);
        tmp_0 = vandq_u8(vshrq_n_u8(register_0, 2), mask);
        tmp_1 = vandq_u8(vshrq_n_u8(register_1, 2), mask);
        tmp_2 = vandq_u8(vshrq_n_u8(register_2, 2), mask);
        tmp_3 = vandq_u8(vshrq_n_u8(register_3, 2), mask);
        tmp_4 = vandq_u8(vshrq_n_u8(register_4, 2), mask);
        tmp_5 = vandq_u8(vshrq_n_u8(register_5, 2), mask);
        tmp_6 = vandq_u8(vshrq_n_u8(register_6, 2), mask);
        tmp_7 = vandq_u8(vshrq_n_u8(register_7, 2), mask);
        vst1q_u8(out + (0 * 1 * 16) + (128 * 3), tmp_0);
        vst1q_u8(out + (1 * 1 * 16) + (128 * 3), tmp_1);
        vst1q_u8(out + (2 * 1 * 16) + (128 * 3), tmp_2);
        vst1q_u8(out + (3 * 1 * 16) + (128 * 3), tmp_3);
        vst1q_u8(out + (4 * 1 * 16) + (128 * 3), tmp_4);
        vst1q_u8(out + (5 * 1 * 16) + (128 * 3), tmp_5);
        vst1q_u8(out + (6 * 1 * 16) + (128 * 3), tmp_6);
        vst1q_u8(out + (7 * 1 * 16) + (128 * 3), tmp_7);
        register_0 = vld1q_u8(in + (0 * 1 * 16) + 384);
        register_1 = vld1q_u8(in + (1 * 1 * 16) + 384);
        register_2 = vld1q_u8(in + (2 * 1 * 16) + 384);
        register_3 = vld1q_u8(in + (3 * 1 * 16) + 384);
        register_4 = vld1q_u8(in + (4 * 1 * 16) + 384);
        register_5 = vld1q_u8(in + (5 * 1 * 16) + 384);
        register_6 = vld1q_u8(in + (6 * 1 * 16) + 384);
        register_7 = vld1q_u8(in + (7 * 1 * 16) + 384);
        tmp_0 = vandq_u8(register_0, mask);
        tmp_1 = vandq_u8(register_1, mask);
        tmp_2 = vandq_u8(register_2, mask);
        tmp_3 = vandq_u8(register_3, mask);
        tmp_4 = vandq_u8(register_4, mask);
        tmp_5 = vandq_u8(register_5, mask);
        tmp_6 = vandq_u8(register_6, mask);
        tmp_7 = vandq_u8(register_7, mask);
        vst1q_u8(out + (0 * 1 * 16) + (128 * 4), tmp_0);
        vst1q_u8(out + (1 * 1 * 16) + (128 * 4), tmp_1);
        vst1q_u8(out + (2 * 1 * 16) + (128 * 4), tmp_2);
        vst1q_u8(out + (3 * 1 * 16) + (128 * 4), tmp_3);
        vst1q_u8(out + (4 * 1 * 16) + (128 * 4), tmp_4);
        vst1q_u8(out + (5 * 1 * 16) + (128 * 4), tmp_5);
        vst1q_u8(out + (6 * 1 * 16) + (128 * 4), tmp_6);
        vst1q_u8(out + (7 * 1 * 16) + (128 * 4), tmp_7);
        tmp_0 = vandq_u8(vshrq_n_u8(register_0, 6), vdupq_n_u8((1ULL << 2) - 1));
        tmp_1 = vandq_u8(vshrq_n_u8(register_1, 6), vdupq_n_u8((1ULL << 2) - 1));
        tmp_2 = vandq_u8(vshrq_n_u8(register_2, 6), vdupq_n_u8((1ULL << 2) - 1));
        tmp_3 = vandq_u8(vshrq_n_u8(register_3, 6), vdupq_n_u8((1ULL << 2) - 1));
        tmp_4 = vandq_u8(vshrq_n_u8(register_4, 6), vdupq_n_u8((1ULL << 2) - 1));
        tmp_5 = vandq_u8(vshrq_n_u8(register_5, 6), vdupq_n_u8((1ULL << 2) - 1));
        tmp_6 = vandq_u8(vshrq_n_u8(register_6, 6), vdupq_n_u8((1ULL << 2) - 1));
        tmp_7 = vandq_u8(vshrq_n_u8(register_7, 6), vdupq_n_u8((1ULL << 2) - 1));
        register_0 = vld1q_u8(in + (0 * 1 * 16) + 512);
        register_1 = vld1q_u8(in + (1 * 1 * 16) + 512);
        register_2 = vld1q_u8(in + (2 * 1 * 16) + 512);
        register_3 = vld1q_u8(in + (3 * 1 * 16) + 512);
        register_4 = vld1q_u8(in + (4 * 1 * 16) + 512);
        register_5 = vld1q_u8(in + (5 * 1 * 16) + 512);
        register_6 = vld1q_u8(in + (6 * 1 * 16) + 512);
        register_7 = vld1q_u8(in + (7 * 1 * 16) + 512);
        tmp_0 = vorrq_u8(vshlq_n_u64(vandq_u8(register_0, vdupq_n_u8((1ULL << 4) - 1)) ,2), tmp_0);
        tmp_1 = vorrq_u8(vshlq_n_u64(vandq_u8(register_1, vdupq_n_u8((1ULL << 4) - 1)) ,2), tmp_1);
        tmp_2 = vorrq_u8(vshlq_n_u64(vandq_u8(register_2, vdupq_n_u8((1ULL << 4) - 1)) ,2), tmp_2);
        tmp_3 = vorrq_u8(vshlq_n_u64(vandq_u8(register_3, vdupq_n_u8((1ULL << 4) - 1)) ,2), tmp_3);
        tmp_4 = vorrq_u8(vshlq_n_u64(vandq_u8(register_4, vdupq_n_u8((1ULL << 4) - 1)) ,2), tmp_4);
        tmp_5 = vorrq_u8(vshlq_n_u64(vandq_u8(register_5, vdupq_n_u8((1ULL << 4) - 1)) ,2), tmp_5);
        tmp_6 = vorrq_u8(vshlq_n_u64(vandq_u8(register_6, vdupq_n_u8((1ULL << 4) - 1)) ,2), tmp_6);
        tmp_7 = vorrq_u8(vshlq_n_u64(vandq_u8(register_7, vdupq_n_u8((1ULL << 4) - 1)) ,2), tmp_7);
        vst1q_u8(out + (0 * 1 * 16) + (128 * 5), tmp_0);
        vst1q_u8(out + (1 * 1 * 16) + (128 * 5), tmp_1);
        vst1q_u8(out + (2 * 1 * 16) + (128 * 5), tmp_2);
        vst1q_u8(out + (3 * 1 * 16) + (128 * 5), tmp_3);
        vst1q_u8(out + (4 * 1 * 16) + (128 * 5), tmp_4);
        vst1q_u8(out + (5 * 1 * 16) + (128 * 5), tmp_5);
        vst1q_u8(out + (6 * 1 * 16) + (128 * 5), tmp_6);
        vst1q_u8(out + (7 * 1 * 16) + (128 * 5), tmp_7);
        tmp_0 = vandq_u8(vshrq_n_u8(register_0, 4), vdupq_n_u8((1ULL << 4) - 1));
        tmp_1 = vandq_u8(vshrq_n_u8(register_1, 4), vdupq_n_u8((1ULL << 4) - 1));
        tmp_2 = vandq_u8(vshrq_n_u8(register_2, 4), vdupq_n_u8((1ULL << 4) - 1));
        tmp_3 = vandq_u8(vshrq_n_u8(register_3, 4), vdupq_n_u8((1ULL << 4) - 1));
        tmp_4 = vandq_u8(vshrq_n_u8(register_4, 4), vdupq_n_u8((1ULL << 4) - 1));
        tmp_5 = vandq_u8(vshrq_n_u8(register_5, 4), vdupq_n_u8((1ULL << 4) - 1));
        tmp_6 = vandq_u8(vshrq_n_u8(register_6, 4), vdupq_n_u8((1ULL << 4) - 1));
        tmp_7 = vandq_u8(vshrq_n_u8(register_7, 4), vdupq_n_u8((1ULL << 4) - 1));
        register_0 = vld1q_u8(in + (0 * 1 * 16) + 640);
        register_1 = vld1q_u8(in + (1 * 1 * 16) + 640);
        register_2 = vld1q_u8(in + (2 * 1 * 16) + 640);
        register_3 = vld1q_u8(in + (3 * 1 * 16) + 640);
        register_4 = vld1q_u8(in + (4 * 1 * 16) + 640);
        register_5 = vld1q_u8(in + (5 * 1 * 16) + 640);
        register_6 = vld1q_u8(in + (6 * 1 * 16) + 640);
        register_7 = vld1q_u8(in + (7 * 1 * 16) + 640);
        tmp_0 = vorrq_u8(vshlq_n_u64(vandq_u8(register_0, vdupq_n_u8((1ULL << 2) - 1)) ,4), tmp_0);
        tmp_1 = vorrq_u8(vshlq_n_u64(vandq_u8(register_1, vdupq_n_u8((1ULL << 2) - 1)) ,4), tmp_1);
        tmp_2 = vorrq_u8(vshlq_n_u64(vandq_u8(register_2, vdupq_n_u8((1ULL << 2) - 1)) ,4), tmp_2);
        tmp_3 = vorrq_u8(vshlq_n_u64(vandq_u8(register_3, vdupq_n_u8((1ULL << 2) - 1)) ,4), tmp_3);
        tmp_4 = vorrq_u8(vshlq_n_u64(vandq_u8(register_4, vdupq_n_u8((1ULL << 2) - 1)) ,4), tmp_4);
        tmp_5 = vorrq_u8(vshlq_n_u64(vandq_u8(register_5, vdupq_n_u8((1ULL << 2) - 1)) ,4), tmp_5);
        tmp_6 = vorrq_u8(vshlq_n_u64(vandq_u8(register_6, vdupq_n_u8((1ULL << 2) - 1)) ,4), tmp_6);
        tmp_7 = vorrq_u8(vshlq_n_u64(vandq_u8(register_7, vdupq_n_u8((1ULL << 2) - 1)) ,4), tmp_7);
        vst1q_u8(out + (0 * 1 * 16) + (128 * 6), tmp_0);
        vst1q_u8(out + (1 * 1 * 16) + (128 * 6), tmp_1);
        vst1q_u8(out + (2 * 1 * 16) + (128 * 6), tmp_2);
        vst1q_u8(out + (3 * 1 * 16) + (128 * 6), tmp_3);
        vst1q_u8(out + (4 * 1 * 16) + (128 * 6), tmp_4);
        vst1q_u8(out + (5 * 1 * 16) + (128 * 6), tmp_5);
        vst1q_u8(out + (6 * 1 * 16) + (128 * 6), tmp_6);
        vst1q_u8(out + (7 * 1 * 16) + (128 * 6), tmp_7);
        tmp_0 = vandq_u8(vshrq_n_u8(register_0, 2), mask);
        tmp_1 = vandq_u8(vshrq_n_u8(register_1, 2), mask);
        tmp_2 = vandq_u8(vshrq_n_u8(register_2, 2), mask);
        tmp_3 = vandq_u8(vshrq_n_u8(register_3, 2), mask);
        tmp_4 = vandq_u8(vshrq_n_u8(register_4, 2), mask);
        tmp_5 = vandq_u8(vshrq_n_u8(register_5, 2), mask);
        tmp_6 = vandq_u8(vshrq_n_u8(register_6, 2), mask);
        tmp_7 = vandq_u8(vshrq_n_u8(register_7, 2), mask);
        vst1q_u8(out + (0 * 1 * 16) + (128 * 7), tmp_0);
        vst1q_u8(out + (1 * 1 * 16) + (128 * 7), tmp_1);
        vst1q_u8(out + (2 * 1 * 16) + (128 * 7), tmp_2);
        vst1q_u8(out + (3 * 1 * 16) + (128 * 7), tmp_3);
        vst1q_u8(out + (4 * 1 * 16) + (128 * 7), tmp_4);
        vst1q_u8(out + (5 * 1 * 16) + (128 * 7), tmp_5);
        vst1q_u8(out + (6 * 1 * 16) + (128 * 7), tmp_6);
        vst1q_u8(out + (7 * 1 * 16) + (128 * 7), tmp_7);

    }

    static void unpack_4bw_8ow_128crw_8uf(const uint8_t *__restrict a_in_p, uint8_t *__restrict a_out_p)
    {
        [[maybe_unused]] auto out = (a_out_p);
        [[maybe_unused]] const auto in = (a_in_p);
        [[maybe_unused]] uint8x16_t register_0;
        [[maybe_unused]] uint8x16_t tmp_0;
        [[maybe_unused]] uint8x16_t register_1;
        [[maybe_unused]] uint8x16_t tmp_1;
        [[maybe_unused]] uint8x16_t register_2;
        [[maybe_unused]] uint8x16_t tmp_2;
        [[maybe_unused]] uint8x16_t register_3;
        [[maybe_unused]] uint8x16_t tmp_3;
        [[maybe_unused]] uint8x16_t register_4;
        [[maybe_unused]] uint8x16_t tmp_4;
        [[maybe_unused]] uint8x16_t register_5;
        [[maybe_unused]] uint8x16_t tmp_5;
        [[maybe_unused]] uint8x16_t register_6;
        [[maybe_unused]] uint8x16_t tmp_6;
        [[maybe_unused]] uint8x16_t register_7;
        [[maybe_unused]] uint8x16_t tmp_7;
        [[maybe_unused]] int8x16_t base_0 = vmovq_n_u8(0ULL);
        [[maybe_unused]] int8x16_t base_1 = vmovq_n_u8(0ULL);
        [[maybe_unused]] int8x16_t base_2 = vmovq_n_u8(0ULL);
        [[maybe_unused]] int8x16_t base_3 = vmovq_n_u8(0ULL);
        [[maybe_unused]] int8x16_t base_4 = vmovq_n_u8(0ULL);
        [[maybe_unused]] int8x16_t base_5 = vmovq_n_u8(0ULL);
        [[maybe_unused]] int8x16_t base_6 = vmovq_n_u8(0ULL);
        [[maybe_unused]] int8x16_t base_7 = vmovq_n_u8(0ULL);
#pragma clang loop unroll(disable)
        for (int i = 0; i < 1; ++i)
        {
            register_0 = vld1q_u8(in + (0 * 1 * 16) + (i * 16) + 0);
            register_1 = vld1q_u8(in + (1 * 1 * 16) + (i * 16) + 0);
            register_2 = vld1q_u8(in + (2 * 1 * 16) + (i * 16) + 0);
            register_3 = vld1q_u8(in + (3 * 1 * 16) + (i * 16) + 0);
            register_4 = vld1q_u8(in + (4 * 1 * 16) + (i * 16) + 0);
            register_5 = vld1q_u8(in + (5 * 1 * 16) + (i * 16) + 0);
            register_6 = vld1q_u8(in + (6 * 1 * 16) + (i * 16) + 0);
            register_7 = vld1q_u8(in + (7 * 1 * 16) + (i * 16) + 0);
            tmp_0 = vandq_u8(register_0, vdupq_n_u8((1ULL << 4) - 1));
            tmp_1 = vandq_u8(register_1, vdupq_n_u8((1ULL << 4) - 1));
            tmp_2 = vandq_u8(register_2, vdupq_n_u8((1ULL << 4) - 1));
            tmp_3 = vandq_u8(register_3, vdupq_n_u8((1ULL << 4) - 1));
            tmp_4 = vandq_u8(register_4, vdupq_n_u8((1ULL << 4) - 1));
            tmp_5 = vandq_u8(register_5, vdupq_n_u8((1ULL << 4) - 1));
            tmp_6 = vandq_u8(register_6, vdupq_n_u8((1ULL << 4) - 1));
            tmp_7 = vandq_u8(register_7, vdupq_n_u8((1ULL << 4) - 1));
            vst1q_u8(out + (i * 16) + (0 * 1 * 16) + (128 * 0), tmp_0);
            vst1q_u8(out + (i * 16) + (1 * 1 * 16) + (128 * 0), tmp_1);
            vst1q_u8(out + (i * 16) + (2 * 1 * 16) + (128 * 0), tmp_2);
            vst1q_u8(out + (i * 16) + (3 * 1 * 16) + (128 * 0), tmp_3);
            vst1q_u8(out + (i * 16) + (4 * 1 * 16) + (128 * 0), tmp_4);
            vst1q_u8(out + (i * 16) + (5 * 1 * 16) + (128 * 0), tmp_5);
            vst1q_u8(out + (i * 16) + (6 * 1 * 16) + (128 * 0), tmp_6);
            vst1q_u8(out + (i * 16) + (7 * 1 * 16) + (128 * 0), tmp_7);
            tmp_0 = vandq_u8(vshrq_n_u8(register_0, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_1 = vandq_u8(vshrq_n_u8(register_1, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_2 = vandq_u8(vshrq_n_u8(register_2, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_3 = vandq_u8(vshrq_n_u8(register_3, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_4 = vandq_u8(vshrq_n_u8(register_4, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_5 = vandq_u8(vshrq_n_u8(register_5, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_6 = vandq_u8(vshrq_n_u8(register_6, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_7 = vandq_u8(vshrq_n_u8(register_7, 4), vdupq_n_u8((1ULL << 4) - 1));
            vst1q_u8(out + (i * 16) + (0 * 1 * 16) + (128 * 1), tmp_0);
            vst1q_u8(out + (i * 16) + (1 * 1 * 16) + (128 * 1), tmp_1);
            vst1q_u8(out + (i * 16) + (2 * 1 * 16) + (128 * 1), tmp_2);
            vst1q_u8(out + (i * 16) + (3 * 1 * 16) + (128 * 1), tmp_3);
            vst1q_u8(out + (i * 16) + (4 * 1 * 16) + (128 * 1), tmp_4);
            vst1q_u8(out + (i * 16) + (5 * 1 * 16) + (128 * 1), tmp_5);
            vst1q_u8(out + (i * 16) + (6 * 1 * 16) + (128 * 1), tmp_6);
            vst1q_u8(out + (i * 16) + (7 * 1 * 16) + (128 * 1), tmp_7);
            register_0 = vld1q_u8(in + (0 * 1 * 16) + (i * 16) + 128);
            register_1 = vld1q_u8(in + (1 * 1 * 16) + (i * 16) + 128);
            register_2 = vld1q_u8(in + (2 * 1 * 16) + (i * 16) + 128);
            register_3 = vld1q_u8(in + (3 * 1 * 16) + (i * 16) + 128);
            register_4 = vld1q_u8(in + (4 * 1 * 16) + (i * 16) + 128);
            register_5 = vld1q_u8(in + (5 * 1 * 16) + (i * 16) + 128);
            register_6 = vld1q_u8(in + (6 * 1 * 16) + (i * 16) + 128);
            register_7 = vld1q_u8(in + (7 * 1 * 16) + (i * 16) + 128);
            tmp_0 = vandq_u8(register_0, vdupq_n_u8((1ULL << 4) - 1));
            tmp_1 = vandq_u8(register_1, vdupq_n_u8((1ULL << 4) - 1));
            tmp_2 = vandq_u8(register_2, vdupq_n_u8((1ULL << 4) - 1));
            tmp_3 = vandq_u8(register_3, vdupq_n_u8((1ULL << 4) - 1));
            tmp_4 = vandq_u8(register_4, vdupq_n_u8((1ULL << 4) - 1));
            tmp_5 = vandq_u8(register_5, vdupq_n_u8((1ULL << 4) - 1));
            tmp_6 = vandq_u8(register_6, vdupq_n_u8((1ULL << 4) - 1));
            tmp_7 = vandq_u8(register_7, vdupq_n_u8((1ULL << 4) - 1));
            vst1q_u8(out + (i * 16) + (0 * 1 * 16) + (128 * 2), tmp_0);
            vst1q_u8(out + (i * 16) + (1 * 1 * 16) + (128 * 2), tmp_1);
            vst1q_u8(out + (i * 16) + (2 * 1 * 16) + (128 * 2), tmp_2);
            vst1q_u8(out + (i * 16) + (3 * 1 * 16) + (128 * 2), tmp_3);
            vst1q_u8(out + (i * 16) + (4 * 1 * 16) + (128 * 2), tmp_4);
            vst1q_u8(out + (i * 16) + (5 * 1 * 16) + (128 * 2), tmp_5);
            vst1q_u8(out + (i * 16) + (6 * 1 * 16) + (128 * 2), tmp_6);
            vst1q_u8(out + (i * 16) + (7 * 1 * 16) + (128 * 2), tmp_7);
            tmp_0 = vandq_u8(vshrq_n_u8(register_0, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_1 = vandq_u8(vshrq_n_u8(register_1, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_2 = vandq_u8(vshrq_n_u8(register_2, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_3 = vandq_u8(vshrq_n_u8(register_3, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_4 = vandq_u8(vshrq_n_u8(register_4, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_5 = vandq_u8(vshrq_n_u8(register_5, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_6 = vandq_u8(vshrq_n_u8(register_6, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_7 = vandq_u8(vshrq_n_u8(register_7, 4), vdupq_n_u8((1ULL << 4) - 1));
            vst1q_u8(out + (i * 16) + (0 * 1 * 16) + (128 * 3), tmp_0);
            vst1q_u8(out + (i * 16) + (1 * 1 * 16) + (128 * 3), tmp_1);
            vst1q_u8(out + (i * 16) + (2 * 1 * 16) + (128 * 3), tmp_2);
            vst1q_u8(out + (i * 16) + (3 * 1 * 16) + (128 * 3), tmp_3);
            vst1q_u8(out + (i * 16) + (4 * 1 * 16) + (128 * 3), tmp_4);
            vst1q_u8(out + (i * 16) + (5 * 1 * 16) + (128 * 3), tmp_5);
            vst1q_u8(out + (i * 16) + (6 * 1 * 16) + (128 * 3), tmp_6);
            vst1q_u8(out + (i * 16) + (7 * 1 * 16) + (128 * 3), tmp_7);
            register_0 = vld1q_u8(in + (0 * 1 * 16) + (i * 16) + 256);
            register_1 = vld1q_u8(in + (1 * 1 * 16) + (i * 16) + 256);
            register_2 = vld1q_u8(in + (2 * 1 * 16) + (i * 16) + 256);
            register_3 = vld1q_u8(in + (3 * 1 * 16) + (i * 16) + 256);
            register_4 = vld1q_u8(in + (4 * 1 * 16) + (i * 16) + 256);
            register_5 = vld1q_u8(in + (5 * 1 * 16) + (i * 16) + 256);
            register_6 = vld1q_u8(in + (6 * 1 * 16) + (i * 16) + 256);
            register_7 = vld1q_u8(in + (7 * 1 * 16) + (i * 16) + 256);
            tmp_0 = vandq_u8(register_0, vdupq_n_u8((1ULL << 4) - 1));
            tmp_1 = vandq_u8(register_1, vdupq_n_u8((1ULL << 4) - 1));
            tmp_2 = vandq_u8(register_2, vdupq_n_u8((1ULL << 4) - 1));
            tmp_3 = vandq_u8(register_3, vdupq_n_u8((1ULL << 4) - 1));
            tmp_4 = vandq_u8(register_4, vdupq_n_u8((1ULL << 4) - 1));
            tmp_5 = vandq_u8(register_5, vdupq_n_u8((1ULL << 4) - 1));
            tmp_6 = vandq_u8(register_6, vdupq_n_u8((1ULL << 4) - 1));
            tmp_7 = vandq_u8(register_7, vdupq_n_u8((1ULL << 4) - 1));
            vst1q_u8(out + (i * 16) + (0 * 1 * 16) + (128 * 4), tmp_0);
            vst1q_u8(out + (i * 16) + (1 * 1 * 16) + (128 * 4), tmp_1);
            vst1q_u8(out + (i * 16) + (2 * 1 * 16) + (128 * 4), tmp_2);
            vst1q_u8(out + (i * 16) + (3 * 1 * 16) + (128 * 4), tmp_3);
            vst1q_u8(out + (i * 16) + (4 * 1 * 16) + (128 * 4), tmp_4);
            vst1q_u8(out + (i * 16) + (5 * 1 * 16) + (128 * 4), tmp_5);
            vst1q_u8(out + (i * 16) + (6 * 1 * 16) + (128 * 4), tmp_6);
            vst1q_u8(out + (i * 16) + (7 * 1 * 16) + (128 * 4), tmp_7);
            tmp_0 = vandq_u8(vshrq_n_u8(register_0, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_1 = vandq_u8(vshrq_n_u8(register_1, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_2 = vandq_u8(vshrq_n_u8(register_2, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_3 = vandq_u8(vshrq_n_u8(register_3, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_4 = vandq_u8(vshrq_n_u8(register_4, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_5 = vandq_u8(vshrq_n_u8(register_5, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_6 = vandq_u8(vshrq_n_u8(register_6, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_7 = vandq_u8(vshrq_n_u8(register_7, 4), vdupq_n_u8((1ULL << 4) - 1));
            vst1q_u8(out + (i * 16) + (0 * 1 * 16) + (128 * 5), tmp_0);
            vst1q_u8(out + (i * 16) + (1 * 1 * 16) + (128 * 5), tmp_1);
            vst1q_u8(out + (i * 16) + (2 * 1 * 16) + (128 * 5), tmp_2);
            vst1q_u8(out + (i * 16) + (3 * 1 * 16) + (128 * 5), tmp_3);
            vst1q_u8(out + (i * 16) + (4 * 1 * 16) + (128 * 5), tmp_4);
            vst1q_u8(out + (i * 16) + (5 * 1 * 16) + (128 * 5), tmp_5);
            vst1q_u8(out + (i * 16) + (6 * 1 * 16) + (128 * 5), tmp_6);
            vst1q_u8(out + (i * 16) + (7 * 1 * 16) + (128 * 5), tmp_7);
            register_0 = vld1q_u8(in + (0 * 1 * 16) + (i * 16) + 384);
            register_1 = vld1q_u8(in + (1 * 1 * 16) + (i * 16) + 384);
            register_2 = vld1q_u8(in + (2 * 1 * 16) + (i * 16) + 384);
            register_3 = vld1q_u8(in + (3 * 1 * 16) + (i * 16) + 384);
            register_4 = vld1q_u8(in + (4 * 1 * 16) + (i * 16) + 384);
            register_5 = vld1q_u8(in + (5 * 1 * 16) + (i * 16) + 384);
            register_6 = vld1q_u8(in + (6 * 1 * 16) + (i * 16) + 384);
            register_7 = vld1q_u8(in + (7 * 1 * 16) + (i * 16) + 384);
            tmp_0 = vandq_u8(register_0, vdupq_n_u8((1ULL << 4) - 1));
            tmp_1 = vandq_u8(register_1, vdupq_n_u8((1ULL << 4) - 1));
            tmp_2 = vandq_u8(register_2, vdupq_n_u8((1ULL << 4) - 1));
            tmp_3 = vandq_u8(register_3, vdupq_n_u8((1ULL << 4) - 1));
            tmp_4 = vandq_u8(register_4, vdupq_n_u8((1ULL << 4) - 1));
            tmp_5 = vandq_u8(register_5, vdupq_n_u8((1ULL << 4) - 1));
            tmp_6 = vandq_u8(register_6, vdupq_n_u8((1ULL << 4) - 1));
            tmp_7 = vandq_u8(register_7, vdupq_n_u8((1ULL << 4) - 1));
            vst1q_u8(out + (i * 16) + (0 * 1 * 16) + (128 * 6), tmp_0);
            vst1q_u8(out + (i * 16) + (1 * 1 * 16) + (128 * 6), tmp_1);
            vst1q_u8(out + (i * 16) + (2 * 1 * 16) + (128 * 6), tmp_2);
            vst1q_u8(out + (i * 16) + (3 * 1 * 16) + (128 * 6), tmp_3);
            vst1q_u8(out + (i * 16) + (4 * 1 * 16) + (128 * 6), tmp_4);
            vst1q_u8(out + (i * 16) + (5 * 1 * 16) + (128 * 6), tmp_5);
            vst1q_u8(out + (i * 16) + (6 * 1 * 16) + (128 * 6), tmp_6);
            vst1q_u8(out + (i * 16) + (7 * 1 * 16) + (128 * 6), tmp_7);
            tmp_0 = vandq_u8(vshrq_n_u8(register_0, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_1 = vandq_u8(vshrq_n_u8(register_1, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_2 = vandq_u8(vshrq_n_u8(register_2, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_3 = vandq_u8(vshrq_n_u8(register_3, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_4 = vandq_u8(vshrq_n_u8(register_4, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_5 = vandq_u8(vshrq_n_u8(register_5, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_6 = vandq_u8(vshrq_n_u8(register_6, 4), vdupq_n_u8((1ULL << 4) - 1));
            tmp_7 = vandq_u8(vshrq_n_u8(register_7, 4), vdupq_n_u8((1ULL << 4) - 1));
            vst1q_u8(out + (i * 16) + (0 * 1 * 16) + (128 * 7), tmp_0);
            vst1q_u8(out + (i * 16) + (1 * 1 * 16) + (128 * 7), tmp_1);
            vst1q_u8(out + (i * 16) + (2 * 1 * 16) + (128 * 7), tmp_2);
            vst1q_u8(out + (i * 16) + (3 * 1 * 16) + (128 * 7), tmp_3);
            vst1q_u8(out + (i * 16) + (4 * 1 * 16) + (128 * 7), tmp_4);
            vst1q_u8(out + (i * 16) + (5 * 1 * 16) + (128 * 7), tmp_5);
            vst1q_u8(out + (i * 16) + (6 * 1 * 16) + (128 * 7), tmp_6);
            vst1q_u8(out + (i * 16) + (7 * 1 * 16) + (128 * 7), tmp_7);
        }
    }

};

};

#endif