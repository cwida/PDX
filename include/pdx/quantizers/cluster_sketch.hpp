#pragma once

#include "pdx/common.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>

namespace PDX {

// 1-bit RaBitQ sketches of a cluster's rows (RaBitQ-Library's split batches): each row's signs of
// its residual to the index's mean, and three estimator factors. A cluster keeps its rows in
// batches of CLUSTER_SKETCH_BATCH_SIZE lanes: every lane's code in FastScan order, then the f_add,
// the f_rescale and the f_error of every lane. A lane with f_add = +inf (a free slot or a
// tombstone) never stops its cluster from being pruned.
class ClusterSketchQuantizer {
  public:
    explicit ClusterSketchQuantizer(uint32_t num_dimensions)
        : num_dimensions(num_dimensions),
          padded_num_dimensions(
              AlignValue<uint32_t, CLUSTER_SKETCH_DIMENSION_ALIGNMENT>(num_dimensions)
          ),
          code_size_in_bytes(padded_num_dimensions / 8),
          row_size_in_bytes(code_size_in_bytes + 3 * sizeof(float)),
          batch_size_in_bytes(row_size_in_bytes * CLUSTER_SKETCH_BATCH_SIZE) {}

    [[nodiscard]] static size_t GetNumBatches(uint32_t num_rows) {
        return (static_cast<size_t>(num_rows) + CLUSTER_SKETCH_BATCH_SIZE - 1) /
               CLUSTER_SKETCH_BATCH_SIZE;
    }

    // embedding and mean: normalized if the index needs it, then rotated. sketch_row: the code
    // (bit 7 of byte i is dimension 8i), then f_add, f_rescale and f_error.
    void EncodeRow(
        const float* PDX_RESTRICT embedding,
        const float* PDX_RESTRICT mean,
        uint8_t* PDX_RESTRICT sketch_row
    ) const {
        std::memset(sketch_row, 0, code_size_in_bytes);
        float squared_norm = 0.0f;
        float absolute_sum = 0.0f;
        float mean_sum_by_sign = 0.0f;
        // TODO(@lkuffo, high): SIMDize this loop per ISA
        for (uint32_t i = 0; i < num_dimensions; ++i) {
            const float residual = embedding[i] - mean[i];
            const bool positive = residual > 0.0f;
            sketch_row[i / 8] |= static_cast<uint8_t>(static_cast<uint8_t>(positive) << (7 - i % 8));
            squared_norm += residual * residual;
            absolute_sum += std::abs(residual);
            mean_sum_by_sign += positive ? mean[i] : -mean[i];
        }
        // <r, b - 1/2> and <mean, b - 1/2>, b being the row's bits; a row equal to the mean gets
        // zero factors, so its estimate is exactly |q - mean|^2 (RaBitQ-Library's corner case)
        float residual_dot_code = 0.5f * absolute_sum;
        const float mean_dot_code = 0.5f * mean_sum_by_sign;
        if (residual_dot_code == 0.0f) {
            residual_dot_code = std::numeric_limits<float>::infinity();
        }
        const float code_squared_norm = 0.25f * static_cast<float>(num_dimensions);
        const float f_add = squared_norm + 2.0f * squared_norm * mean_dot_code / residual_dot_code;
        const float f_rescale = -2.0f * squared_norm / residual_dot_code;
        // The query multiplies in CLUSTER_SKETCH_EPSILON; the clamp undoes float rounding
        const float f_error =
            2.0f * std::sqrt(squared_norm) *
            std::sqrt(
                std::max(
                    0.0f,
                    squared_norm * code_squared_norm / (residual_dot_code * residual_dot_code) -
                        1.0f
                ) /
                static_cast<float>(num_dimensions - 1)
            );
        float* factors = reinterpret_cast<float*>(sketch_row + code_size_in_bytes);
        std::memcpy(factors, &f_add, sizeof(float));
        std::memcpy(factors + 1, &f_rescale, sizeof(float));
        std::memcpy(factors + 2, &f_error, sizeof(float));
    }

    // FastScan order (RaBitQ-Library's pack_codes): code byte i of the batch's rows is a column of
    // CLUSTER_SKETCH_BATCH_SIZE bytes, its upper nibbles in the first half and its lower nibbles
    // in the second; in each half, lanes 0 to 15 take the low nibbles of the bytes and lanes 16 to
    // 31 the high ones, byte j holding lane kPerm0[j] (0, 8, 1, 9, ..., 7, 15) and lane kPerm0[j] + 16
    void StoreRowInBatch(
        const uint8_t* PDX_RESTRICT sketch_row,
        uint8_t* PDX_RESTRICT batch,
        uint32_t lane
    ) const {
        constexpr uint32_t HALF_COLUMN = CLUSTER_SKETCH_BATCH_SIZE / 2;
        const uint32_t position = 2 * (lane % 8) + (lane % HALF_COLUMN) / 8;
        const uint32_t shift = lane < HALF_COLUMN ? 0 : 4;
        const auto kept_nibble = static_cast<uint8_t>(lane < HALF_COLUMN ? 0xF0 : 0x0F);
        for (uint32_t i = 0; i < code_size_in_bytes; ++i) {
            uint8_t* column = batch + static_cast<size_t>(i) * CLUSTER_SKETCH_BATCH_SIZE;
            const uint8_t code = sketch_row[i];
            column[position] =
                static_cast<uint8_t>((column[position] & kept_nibble) | ((code >> 4) << shift));
            column[HALF_COLUMN + position] = static_cast<uint8_t>(
                (column[HALF_COLUMN + position] & kept_nibble) | ((code & 0x0F) << shift)
            );
        }
        uint8_t* factors = batch + static_cast<size_t>(code_size_in_bytes) * CLUSTER_SKETCH_BATCH_SIZE;
        for (uint32_t factor = 0; factor < 3; ++factor) {
            std::memcpy(
                factors + (factor * CLUSTER_SKETCH_BATCH_SIZE + lane) * sizeof(float),
                sketch_row + code_size_in_bytes + factor * sizeof(float),
                sizeof(float)
            );
        }
    }

    void ReadRowFromBatch(
        const uint8_t* PDX_RESTRICT batch,
        uint32_t lane,
        uint8_t* PDX_RESTRICT sketch_row
    ) const {
        constexpr uint32_t HALF_COLUMN = CLUSTER_SKETCH_BATCH_SIZE / 2;
        const uint32_t position = 2 * (lane % 8) + (lane % HALF_COLUMN) / 8;
        const uint32_t shift = lane < HALF_COLUMN ? 0 : 4;
        for (uint32_t i = 0; i < code_size_in_bytes; ++i) {
            const uint8_t* column = batch + static_cast<size_t>(i) * CLUSTER_SKETCH_BATCH_SIZE;
            const uint32_t upper = (column[position] >> shift) & 0x0F;
            const uint32_t lower = (column[HALF_COLUMN + position] >> shift) & 0x0F;
            sketch_row[i] = static_cast<uint8_t>((upper << 4) | lower);
        }
        const uint8_t* factors =
            batch + static_cast<size_t>(code_size_in_bytes) * CLUSTER_SKETCH_BATCH_SIZE;
        for (uint32_t factor = 0; factor < 3; ++factor) {
            std::memcpy(
                sketch_row + code_size_in_bytes + factor * sizeof(float),
                factors + (factor * CLUSTER_SKETCH_BATCH_SIZE + lane) * sizeof(float),
                sizeof(float)
            );
        }
    }

    void ClearLaneInBatch(uint8_t* batch, uint32_t lane) const {
        float* lane_factors = reinterpret_cast<float*>(
                                  batch + static_cast<size_t>(code_size_in_bytes) *
                                              CLUSTER_SKETCH_BATCH_SIZE
                              ) +
                              lane;
        lane_factors[0] = std::numeric_limits<float>::infinity();
        lane_factors[CLUSTER_SKETCH_BATCH_SIZE] = 0.0f;
        lane_factors[2 * CLUSTER_SKETCH_BATCH_SIZE] = 0.0f;
    }

    // The codes are zeroed too: the last batch of a saved cluster carries its free lanes
    void ClearBatches(uint8_t* batches, size_t num_batches) const {
        std::memset(batches, 0, num_batches * batch_size_in_bytes);
        for (size_t batch = 0; batch < num_batches; ++batch) {
            float* f_add = reinterpret_cast<float*>(
                batches + batch * batch_size_in_bytes +
                static_cast<size_t>(code_size_in_bytes) * CLUSTER_SKETCH_BATCH_SIZE
            );
            std::fill_n(f_add, CLUSTER_SKETCH_BATCH_SIZE, std::numeric_limits<float>::infinity());
        }
    }

    const uint32_t num_dimensions;
    const uint32_t padded_num_dimensions;
    const uint32_t code_size_in_bytes;
    const size_t row_size_in_bytes;
    const size_t batch_size_in_bytes;
};

// A query's side of the sketch estimate (RaBitQ-Library's SplitBatchQuery, high-accuracy table).
// The lookup table holds, per 4 dimensions, the 16 sums of the query values a nibble's bits pick,
// as u16 split into low and high bytes; per group of 4 such codebooks, the 64 low bytes, then the
// 64 high bytes.
class ClusterSketchQuery {
  public:
    static constexpr uint32_t NIBBLE_VALUES = 16;
    static constexpr uint32_t CODEBOOKS_PER_GROUP = 4;

    // transformed_query and mean: normalized if the index needs it, then rotated
    ClusterSketchQuery(
        const float* PDX_RESTRICT transformed_query,
        const float* PDX_RESTRICT mean,
        const ClusterSketchQuantizer& quantizer
    )
        : num_codebooks(quantizer.padded_num_dimensions / 4),
          lookup_table(new uint8_t[static_cast<size_t>(num_codebooks) * NIBBLE_VALUES * 2]) {
        const uint32_t num_dimensions = quantizer.num_dimensions;
        std::unique_ptr<float[]> padded_query(new float[quantizer.padded_num_dimensions]());
        std::copy_n(transformed_query, num_dimensions, padded_query.get());
        float query_sum = 0.0f;
        squared_distance_to_mean = 0.0f;
        for (uint32_t i = 0; i < num_dimensions; ++i) {
            const float difference = transformed_query[i] - mean[i];
            squared_distance_to_mean += difference * difference;
            query_sum += transformed_query[i];
        }
        // Turns <b, q> into <b - 1/2, q>, b being a row's bits
        centering_correction = -0.5f * query_sum;
        error_scale = CLUSTER_SKETCH_EPSILON * std::sqrt(squared_distance_to_mean);

        const size_t num_entries = static_cast<size_t>(num_codebooks) * NIBBLE_VALUES;
        std::unique_ptr<float[]> entries(new float[num_entries]);
        for (uint32_t codebook = 0; codebook < num_codebooks; ++codebook) {
            const float* values = padded_query.get() + static_cast<size_t>(codebook) * 4;
            for (uint32_t nibble = 0; nibble < NIBBLE_VALUES; ++nibble) {
                entries[static_cast<size_t>(codebook) * NIBBLE_VALUES + nibble] =
                    ((nibble & 8) ? values[0] : 0.0f) + ((nibble & 4) ? values[1] : 0.0f) +
                    ((nibble & 2) ? values[2] : 0.0f) + ((nibble & 1) ? values[3] : 0.0f);
            }
        }
        const auto [lowest, highest] = std::minmax_element(entries.get(), entries.get() + num_entries);
        const float lowest_entry = *lowest;
        lookup_table_step = (*highest - lowest_entry) / 65535.0f;
        lookup_table_base_sum = lowest_entry * static_cast<float>(num_codebooks);
        const float inverse_step = lookup_table_step > 0.0f ? 1.0f / lookup_table_step : 0.0f;
        for (uint32_t codebook = 0; codebook < num_codebooks; ++codebook) {
            const size_t group = codebook / CODEBOOKS_PER_GROUP;
            uint8_t* low_bytes = lookup_table.get() +
                                 group * CODEBOOKS_PER_GROUP * NIBBLE_VALUES * 2 +
                                 (codebook % CODEBOOKS_PER_GROUP) * NIBBLE_VALUES;
            uint8_t* high_bytes = low_bytes + CODEBOOKS_PER_GROUP * NIBBLE_VALUES;
            for (uint32_t nibble = 0; nibble < NIBBLE_VALUES; ++nibble) {
                const float entry = entries[static_cast<size_t>(codebook) * NIBBLE_VALUES + nibble];
                const auto quantized =
                    static_cast<uint16_t>(std::lround((entry - lowest_entry) * inverse_step));
                low_bytes[nibble] = static_cast<uint8_t>(quantized & 0xFF);
                high_bytes[nibble] = static_cast<uint8_t>(quantized >> 8);
            }
        }
    }

    const uint32_t num_codebooks;
    std::unique_ptr<uint8_t[]> lookup_table;
    float lookup_table_step;
    float lookup_table_base_sum;
    float centering_correction;
    float squared_distance_to_mean;
    float error_scale;
};

} // namespace PDX
