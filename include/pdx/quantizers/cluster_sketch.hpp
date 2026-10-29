#pragma once

#include "pdx/common.hpp"
#include "pdx/distance_computers/base_computers.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>

namespace PDX {

// 1-bit RaBitQ sketches of a cluster's rows (RaBitQ-Library's split batches): each row's signs of
// its residual to the index's mean, and three estimator factors. At d = 64, a row's sketch row
// (EncodeRow's output, which splits and merges move between clusters) is 20 bytes, bit 7 of code
// byte i being dimension 8i:
//
//            [ dims 0-7 ][ dims 8-15 ] ... [ dims 56-63 ][ f_add ][ f_rescale ][ f_error ]
//   byte      0           1                 7             8        12           16        20
//
// A cluster stores its slots in batches of CLUSTER_SKETCH_BATCH_SIZE lanes, which the scan reads
// together: slot s is lane s % 32 of batch s / 32. A batch holds one 32-byte column per code byte
// (code byte i of every lane), then each factor of every lane, as floats in lane order:
//
//            [ col 0 ][ col 1 ] ... [ col 7 ][ 32 f_add ][ 32 f_rescale ][ 32 f_error ]
//   offset    0        32            224      256         384             512          640
//
// In column i, the first 16 bytes hold the upper nibbles of the lanes' code byte i (dimensions 8i
// to 8i + 3) and the last 16 bytes its lower nibbles. Each byte holds two lanes, "a,b" meaning
// lane a in the low nibble and lane b in the high one:
//
//   byte     0     1     2     3          14    15     16    17         31
//   lanes    0,16  8,24  1,17  9,25  ...  7,23  15,31  0,16  8,24  ...  15,31
//            \___ upper nibbles ____________________/  \__ lower nibbles ___/
//
// Byte j of a half holds lane CLUSTER_SKETCH_LANE_AT_POSITION[j]. The scan looks up 16 lanes'
// nibbles in the query's table with one 16-byte shuffle; this order makes the even bytes lanes 0-7
// and the odd bytes lanes 8-15, which the scan separates when it adds the bytes as u16 words
// (RaBitQ-Library's FastScan order). A lane with f_add = +inf (a free slot or a tombstone) never
// stops its cluster from being pruned.
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

    // Where the batch holding slot index starts (its lane is index % CLUSTER_SKETCH_BATCH_SIZE)
    [[nodiscard]] size_t GetBatchOffsetInBytes(size_t index) const {
        return (index / CLUSTER_SKETCH_BATCH_SIZE) * batch_size_in_bytes;
    }

    // The mean of up to CLUSTER_SKETCH_MEAN_SAMPLE_SIZE evenly spaced rows (as SAQ's fit)
    [[nodiscard]] static std::unique_ptr<float[]> ComputeMean(
        const float* embeddings,
        size_t num_embeddings,
        uint32_t num_dimensions
    ) {
        const size_t num_samples = std::min(num_embeddings, CLUSTER_SKETCH_MEAN_SAMPLE_SIZE);
        std::unique_ptr<double[]> mean_sum(new double[num_dimensions]());
        for (size_t sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
            const float* row =
                embeddings + (sample_idx * num_embeddings / num_samples) * num_dimensions;
            for (uint32_t j = 0; j < num_dimensions; ++j) {
                mean_sum[j] += row[j];
            }
        }
        const double inverse_num_samples = 1.0 / static_cast<double>(num_samples);
        std::unique_ptr<float[]> mean(new float[num_dimensions]);
        for (uint32_t j = 0; j < num_dimensions; ++j) {
            mean[j] = static_cast<float>(mean_sum[j] * inverse_num_samples);
        }
        return mean;
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
        DistanceComputer<DistanceMetric::L2SQ, Quantization::F32>::ComputeSketchCodeAndSums(
            embedding,
            mean,
            num_dimensions,
            sketch_row,
            squared_norm,
            absolute_sum,
            mean_sum_by_sign
        );
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
    // 31 the high ones, byte j holding lane kPerm0[j] (0, 8, 1, 9, ..., 7, 15) and lane kPerm0[j] +
    // 16
    void StoreRowInBatch(
        const uint8_t* PDX_RESTRICT sketch_row,
        uint8_t* PDX_RESTRICT batch,
        uint32_t lane
    ) const {
        DistanceComputer<DistanceMetric::L2SQ, Quantization::F32>::StoreSketchLane(
            sketch_row, code_size_in_bytes, batch, lane
        );
        uint8_t* factors =
            batch + static_cast<size_t>(code_size_in_bytes) * CLUSTER_SKETCH_BATCH_SIZE;
        for (uint32_t factor = 0; factor < 3; ++factor) {
            std::memcpy(
                factors + (factor * CLUSTER_SKETCH_BATCH_SIZE + lane) * sizeof(float),
                sketch_row + code_size_in_bytes + factor * sizeof(float),
                sizeof(float)
            );
        }
    }

    // EncodeRow's rows into batches from lane 0: every CLUSTER_SKETCH_TRANSPOSE_COLUMNS code bytes
    // of a batch's rows in one transpose, read in place. A batch short of rows, and a chunk whose
    // loads would pass the end of a row, go through a zero-padded tile
    void StoreRowsInBatches(
        const uint8_t* PDX_RESTRICT sketch_rows,
        size_t num_rows,
        uint8_t* PDX_RESTRICT batches
    ) const {
        using computer = DistanceComputer<DistanceMetric::L2SQ, Quantization::F32>;
        alignas(16) uint8_t tile[CLUSTER_SKETCH_BATCH_SIZE * CLUSTER_SKETCH_TRANSPOSE_COLUMNS];
        for (size_t first_row = 0; first_row < num_rows; first_row += CLUSTER_SKETCH_BATCH_SIZE) {
            const size_t batch_rows = std::min(CLUSTER_SKETCH_BATCH_SIZE, num_rows - first_row);
            const uint8_t* rows = sketch_rows + first_row * row_size_in_bytes;
            uint8_t* batch = batches + GetBatchOffsetInBytes(first_row);
            for (size_t first_column = 0; first_column < code_size_in_bytes;
                 first_column += CLUSTER_SKETCH_TRANSPOSE_COLUMNS) {
                const size_t num_columns =
                    std::min(CLUSTER_SKETCH_TRANSPOSE_COLUMNS, code_size_in_bytes - first_column);
                if (batch_rows == CLUSTER_SKETCH_BATCH_SIZE &&
                    first_column + CLUSTER_SKETCH_TRANSPOSE_COLUMNS <= row_size_in_bytes) {
                    computer::PackSketchColumns(
                        rows, row_size_in_bytes, first_column, num_columns, batch
                    );
                    continue;
                }
                std::memset(tile, 0, sizeof(tile));
                for (size_t row = 0; row < batch_rows; ++row) {
                    std::memcpy(
                        tile + row * CLUSTER_SKETCH_TRANSPOSE_COLUMNS,
                        rows + row * row_size_in_bytes + first_column,
                        num_columns
                    );
                }
                // The tile holds this chunk's columns only
                computer::PackSketchColumns(
                    tile,
                    CLUSTER_SKETCH_TRANSPOSE_COLUMNS,
                    0,
                    num_columns,
                    batch + first_column * CLUSTER_SKETCH_BATCH_SIZE
                );
            }
            float* f_add = reinterpret_cast<float*>(
                batch + static_cast<size_t>(code_size_in_bytes) * CLUSTER_SKETCH_BATCH_SIZE
            );
            float* f_rescale = f_add + CLUSTER_SKETCH_BATCH_SIZE;
            float* f_error = f_rescale + CLUSTER_SKETCH_BATCH_SIZE;
            for (size_t lane = 0; lane < batch_rows; ++lane) {
                float factors[3];
                std::memcpy(
                    factors, rows + lane * row_size_in_bytes + code_size_in_bytes, sizeof(factors)
                );
                f_add[lane] = factors[0];
                f_rescale[lane] = factors[1];
                f_error[lane] = factors[2];
            }
            for (size_t lane = batch_rows; lane < CLUSTER_SKETCH_BATCH_SIZE; ++lane) {
                ClearLaneInBatch(batch, static_cast<uint32_t>(lane));
            }
        }
    }

    void ReadRowFromBatch(
        const uint8_t* PDX_RESTRICT batch,
        uint32_t lane,
        uint8_t* PDX_RESTRICT sketch_row
    ) const {
        DistanceComputer<DistanceMetric::L2SQ, Quantization::F32>::ReadSketchLane(
            batch, code_size_in_bytes, lane, sketch_row
        );
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

    // The inverse of StoreRowsInBatches: num_rows rows from lane 0
    void ReadRowsFromBatches(
        const uint8_t* PDX_RESTRICT batches,
        size_t num_rows,
        uint8_t* PDX_RESTRICT sketch_rows
    ) const {
        using computer = DistanceComputer<DistanceMetric::L2SQ, Quantization::F32>;
        for (size_t first_row = 0; first_row < num_rows; first_row += CLUSTER_SKETCH_BATCH_SIZE) {
            const size_t batch_rows = std::min(CLUSTER_SKETCH_BATCH_SIZE, num_rows - first_row);
            const uint8_t* batch = batches + GetBatchOffsetInBytes(first_row);
            uint8_t* rows = sketch_rows + first_row * row_size_in_bytes;
            for (size_t first_column = 0; first_column < code_size_in_bytes;
                 first_column += CLUSTER_SKETCH_TRANSPOSE_COLUMNS) {
                computer::UnpackSketchColumns(
                    batch,
                    first_column,
                    std::min(CLUSTER_SKETCH_TRANSPOSE_COLUMNS, code_size_in_bytes - first_column),
                    batch_rows,
                    rows,
                    row_size_in_bytes
                );
            }
            const auto* f_add = reinterpret_cast<const float*>(
                batch + static_cast<size_t>(code_size_in_bytes) * CLUSTER_SKETCH_BATCH_SIZE
            );
            const float* f_rescale = f_add + CLUSTER_SKETCH_BATCH_SIZE;
            const float* f_error = f_rescale + CLUSTER_SKETCH_BATCH_SIZE;
            for (size_t lane = 0; lane < batch_rows; ++lane) {
                const float factors[3] = {f_add[lane], f_rescale[lane], f_error[lane]};
                std::memcpy(
                    rows + lane * row_size_in_bytes + code_size_in_bytes, factors, sizeof(factors)
                );
            }
        }
    }

    // The two lanes may share a batch
    void CopyLaneBetweenBatches(
        const uint8_t* source_batch,
        uint32_t source_lane,
        uint8_t* target_batch,
        uint32_t target_lane
    ) const {
        DistanceComputer<DistanceMetric::L2SQ, Quantization::F32>::CopySketchLane(
            source_batch, source_lane, target_batch, target_lane, code_size_in_bytes
        );
        const size_t factors = static_cast<size_t>(code_size_in_bytes) * CLUSTER_SKETCH_BATCH_SIZE;
        for (uint32_t factor = 0; factor < 3; ++factor) {
            std::memcpy(
                target_batch + factors +
                    (factor * CLUSTER_SKETCH_BATCH_SIZE + target_lane) * sizeof(float),
                source_batch + factors +
                    (factor * CLUSTER_SKETCH_BATCH_SIZE + source_lane) * sizeof(float),
                sizeof(float)
            );
        }
    }

    void ClearLaneInBatch(uint8_t* batch, uint32_t lane) const {
        float* lane_factors =
            reinterpret_cast<float*>(
                batch + static_cast<size_t>(code_size_in_bytes) * CLUSTER_SKETCH_BATCH_SIZE
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

// The query's side of the sketch estimate that no index's mean enters (RaBitQ-Library's
// SplitBatchQuery, high-accuracy table): f_add carries each row's term of the mean, so one table
// serves indexes with different means. The lookup table holds, per 4 dimensions, the 16 sums of the
// query values a nibble's bits pick, as u16 split into low and high bytes; per group of 4 such
// codebooks, the 64 low bytes, then the 64 high bytes.
// For DuckIR, where one query checks the sketches of many indexes (one per row group, sharing the
// rotation): build it once per query from the transformed query and pass it to every index's
// BeginIterativeSearch*, so the table is built once instead of once per index.
class ClusterSketchQueryTable {
  public:
    static constexpr size_t NIBBLE_VALUES = 16;
    static constexpr size_t CODEBOOKS_PER_GROUP = 4;

    // transformed_query: normalized if the index needs it, then rotated
    ClusterSketchQueryTable(const float* PDX_RESTRICT transformed_query, uint32_t num_dimensions)
        : num_dimensions(num_dimensions),
          num_codebooks(
              AlignValue<uint32_t, CLUSTER_SKETCH_DIMENSION_ALIGNMENT>(num_dimensions) / 4
          ),
          lookup_table(new uint8_t[static_cast<size_t>(num_codebooks) * NIBBLE_VALUES * 2]) {
        std::unique_ptr<float[]> padded_query(new float[static_cast<size_t>(num_codebooks) * 4]());
        std::copy_n(transformed_query, num_dimensions, padded_query.get());
        float query_sum = 0.0f;
        for (uint32_t i = 0; i < num_dimensions; ++i) {
            query_sum += transformed_query[i];
        }
        // Turns <b, q> into <b - 1/2, q>, b being a row's bits
        centering_correction = -0.5f * query_sum;

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
        const auto [lowest, highest] =
            std::minmax_element(entries.get(), entries.get() + num_entries);
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

    const uint32_t num_dimensions;
    const uint32_t num_codebooks;
    std::unique_ptr<uint8_t[]> lookup_table;
    float lookup_table_step;
    float lookup_table_base_sum;
    float centering_correction;
};

// A query's side of the sketch estimate for one index: the table, and the query's distance to the
// index's mean
class ClusterSketchQuery {
  public:
    // transformed_query and mean: normalized if the index needs it, then rotated. shared_table: the
    // caller's table of this query, of the index's dimensions (nullptr: built here)
    ClusterSketchQuery(
        const float* PDX_RESTRICT transformed_query,
        const float* PDX_RESTRICT mean,
        const ClusterSketchQuantizer& quantizer,
        const ClusterSketchQueryTable* shared_table = nullptr
    )
        : owned_table(
              shared_table != nullptr ? nullptr
                                      : std::make_unique<ClusterSketchQueryTable>(
                                            transformed_query,
                                            quantizer.num_dimensions
                                        )
          ),
          table(shared_table != nullptr ? *shared_table : *owned_table) {
        assert(table.num_dimensions == quantizer.num_dimensions);
        for (uint32_t i = 0; i < quantizer.num_dimensions; ++i) {
            const float difference = transformed_query[i] - mean[i];
            squared_distance_to_mean += difference * difference;
        }
        error_scale = CLUSTER_SKETCH_EPSILON * std::sqrt(squared_distance_to_mean);
    }

    std::unique_ptr<ClusterSketchQueryTable> owned_table;
    const ClusterSketchQueryTable& table;
    float squared_distance_to_mean = 0.0f;
    float error_scale;
};

} // namespace PDX
