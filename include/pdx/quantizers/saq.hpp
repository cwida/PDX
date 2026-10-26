#pragma once

#include "pdx/common.hpp"
#include "pdx/distance_computers/base_computers.hpp"
#include "pdx/quantizers/scalar.hpp"
#include "pdx/utils.hpp"
#include "superkmeans/common.h"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace PDX {

struct SAQSegment {
    uint32_t start;
    uint32_t end;
    uint32_t width;
    float scale;
};

// An encoded embedding, e.g. d = 384 at 3 bits with chunk widths 8, 3, 3, 1, 1, 1 in 3 segments
// (chunk 0, chunks 1-2, chunks 3-5):
//
//   chunk     0 (head)   1          2          3        4        5
//   width     8          3          3          1        1        1
//            [ 64 B     ][ 24 B    ][ 24 B    ][ 8 B   ][ 8 B   ][ 8 B   ][ r0 r1 r2 |r|^2 ]
//   offset    0          64         88         112      120      128      136              144
//            \________________________ records ______________________/ \___ fp16 factors __/
//
// chunk_record_offsets = {0, 64, 88, 112, 120, 128, 136}. The factors are each segment's rescale,
// then the residual's squared norm. A chunk's record holds its 64 levels in 8 * width bytes
// (ScalarComputer<L2SQ, SAQ>::PackRecord): as they are at 8 bits, else split into parts of 4, 2 and
// 1 bits, low bits first. For example, width 3 = 2 + 1:
//
//   bytes 0-15    2-bit part: byte b holds dims b, b + 16, b + 32, b + 48 in bits 0-1 to 6-7
//   bytes 16-23   1-bit part: bit j holds bit 2 of dim j
//
// and a 4-bit part (widths 4 to 7) is 32 bytes, byte b holding dim b and dim b + 32 in its nibbles
struct SAQModel {
    uint32_t num_dimensions = 0;
    uint32_t bits_per_dimension = 0;
    std::unique_ptr<float[]> rotated_mean;
    // Unit of quantization
    uint32_t num_segments = 0;
    std::unique_ptr<SAQSegment[]> segments;
    // Unit of scan, what PDX works with
    uint32_t num_chunks = 0;
    std::unique_ptr<uint32_t[]> chunk_widths;
    std::unique_ptr<uint32_t[]> chunk_segments;
    std::unique_ptr<float[]> chunk_code_centers;
    std::unique_ptr<float[]> chunk_remaining_variances;
    // Where each chunk's record starts in an encoded embedding; the last entry is the records' size
    std::unique_ptr<size_t[]> chunk_record_offsets;

    SAQModel() = default;
    SAQModel(SAQModel&&) = default;
    SAQModel& operator=(SAQModel&&) = default;

    SAQModel(const SAQModel& other)
        : num_dimensions(other.num_dimensions), bits_per_dimension(other.bits_per_dimension),
          rotated_mean(CopyArray(other.rotated_mean, other.num_dimensions)),
          num_segments(other.num_segments), segments(CopyArray(other.segments, other.num_segments)),
          chunk_remaining_variances(CopyArray(other.chunk_remaining_variances, other.num_chunks)) {
        ComputeChunks();
    }

    SAQModel& operator=(const SAQModel& other) { return *this = SAQModel(other); }

    // The chunks' records, then the segments' rescales and the residual's squared norm in fp16
    size_t GetEncodedEmbeddingSize() const {
        return chunk_record_offsets[num_chunks] + sizeof(uint16_t) * (num_segments + 1);
    }

    // An 8-bit head chunk is kept in the PDX U8 layout for the head kernels; at any other width it
    // is a record like the tail chunks
    bool HasPDXU8Head() const { return chunk_widths[0] == SAQ_MAX_CODE_WIDTH; }

    size_t GetInMemorySizeInBytes() const {
        return sizeof(*this) + num_dimensions * sizeof(float) + num_segments * sizeof(SAQSegment) +
               num_chunks * (2 * sizeof(uint32_t) + 2 * sizeof(float)) +
               (num_chunks + 1) * sizeof(size_t);
    }

    void ComputeChunks() {
        num_chunks = (num_dimensions + SAQ_CHUNK_SIZE - 1) / SAQ_CHUNK_SIZE;
        chunk_widths.reset(new uint32_t[num_chunks]);
        chunk_segments.reset(new uint32_t[num_chunks]);
        chunk_code_centers.reset(new float[num_chunks]);
        chunk_record_offsets.reset(new size_t[num_chunks + 1]);
        chunk_record_offsets[0] = 0;
        for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            for (uint32_t segment_idx = 0; segment_idx < num_segments; ++segment_idx) {
                const SAQSegment& segment = segments[segment_idx];
                if (chunk_idx * SAQ_CHUNK_SIZE >= segment.start &&
                    chunk_idx * SAQ_CHUNK_SIZE < segment.end) {
                    chunk_widths[chunk_idx] = segment.width;
                    chunk_segments[chunk_idx] = segment_idx;
                }
            }
            chunk_code_centers[chunk_idx] =
                0.5f * static_cast<float>((1u << chunk_widths[chunk_idx]) - 1);
            chunk_record_offsets[chunk_idx + 1] =
                chunk_record_offsets[chunk_idx] + size_t{8} * chunk_widths[chunk_idx];
        }
    }

    void Save(std::ostream& out) const {
        WriteValue(out, num_dimensions);
        WriteValue(out, bits_per_dimension);
        WriteValue(out, num_segments);
        out.write(
            reinterpret_cast<const char*>(rotated_mean.get()),
            static_cast<std::streamsize>(num_dimensions * sizeof(float))
        );
        for (uint32_t segment_idx = 0; segment_idx < num_segments; ++segment_idx) {
            WriteValue(out, segments[segment_idx].start);
            WriteValue(out, segments[segment_idx].end);
            WriteValue(out, segments[segment_idx].width);
            WriteValue(out, segments[segment_idx].scale);
        }
        out.write(
            reinterpret_cast<const char*>(chunk_remaining_variances.get()),
            static_cast<std::streamsize>(num_chunks * sizeof(float))
        );
    }

    template <class Reader>
    void Load(Reader& reader) {
        num_dimensions = ReadValue<uint32_t>(reader);
        bits_per_dimension = ReadValue<uint32_t>(reader);
        num_segments = ReadValue<uint32_t>(reader);
        rotated_mean.reset(new float[num_dimensions]);
        reader.Read(rotated_mean.get(), num_dimensions * sizeof(float));
        segments.reset(new SAQSegment[num_segments]);
        for (uint32_t segment_idx = 0; segment_idx < num_segments; ++segment_idx) {
            segments[segment_idx].start = ReadValue<uint32_t>(reader);
            segments[segment_idx].end = ReadValue<uint32_t>(reader);
            segments[segment_idx].width = ReadValue<uint32_t>(reader);
            segments[segment_idx].scale = ReadValue<float>(reader);
        }
        ComputeChunks();
        chunk_remaining_variances.reset(new float[num_chunks]);
        reader.Read(chunk_remaining_variances.get(), num_chunks * sizeof(float));
    }
};

struct SAQFit {
    skmeans::AllocatedBuffer<float> rotation;
    SAQModel model;
};

class SAQEncodingScratch {
  public:
    explicit SAQEncodingScratch(uint32_t num_dimensions)
        : residual(new float[num_dimensions]),
          levels_by_chunk(new uint8_t[AlignValue<uint32_t, SAQ_CHUNK_SIZE>(num_dimensions)]()) {}

    std::unique_ptr<float[]> residual;
    // The levels of whole chunks; the last chunk's levels past the dimensions stay zero
    std::unique_ptr<uint8_t[]> levels_by_chunk;
};

class SAQQuantizer : public Quantizer {
  public:
    explicit SAQQuantizer(size_t num_dimensions) : Quantizer(num_dimensions) {}

    static SAQFit Fit(
        const float* PDX_RESTRICT embeddings,
        size_t num_embeddings,
        uint32_t num_dimensions,
        uint32_t bits_per_dimension,
        uint32_t seed,
        ParallelExecutor& executor,
        MemoryAllocator& allocator
    ) {
        if (num_dimensions < SAQ_MIN_DIMENSIONS) {
            throw std::invalid_argument(
                "SAQ needs at least " + std::to_string(SAQ_MIN_DIMENSIONS) + " dimensions, got " +
                std::to_string(num_dimensions)
            );
        }
        const uint32_t d = num_dimensions;
        const size_t num_samples = std::min(num_embeddings, SAQ_FIT_SAMPLE_SIZE);
        SAQFit fit;
        SAQModel& model = fit.model;
        model.num_dimensions = d;
        model.bits_per_dimension = bits_per_dimension;

        std::unique_ptr<double[]> mean_sum(new double[d]());
        for (size_t sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
            const float* row =
                embeddings + GetSampleRowIndex(sample_idx, num_samples, num_embeddings) * d;
            for (uint32_t j = 0; j < d; ++j) {
                mean_sum[j] += row[j];
            }
        }
        std::unique_ptr<float[]> mean(new float[d]);
        for (uint32_t j = 0; j < d; ++j) {
            mean[j] = static_cast<float>(mean_sum[j] / static_cast<double>(num_samples));
        }
        const auto covariance = ComputeCovariance(
            embeddings, num_embeddings, num_samples, d, mean.get(), executor, allocator
        );
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(
            Eigen::Map<const Eigen::MatrixXf>(covariance.get(), d, d)
        );
        // The eigenpairs by decreasing eigenvalue
        std::unique_ptr<double[]> eigenvalues(new double[d]);
        auto basis_values = skmeans::AllocateBuffer<float>(allocator, static_cast<size_t>(d) * d);
        Eigen::Map<Eigen::MatrixXf> basis(basis_values.get(), d, d);
        for (uint32_t j = 0; j < d; ++j) {
            eigenvalues[j] = std::max(0.0, static_cast<double>(solver.eigenvalues()(d - 1 - j)));
            basis.col(j) = solver.eigenvectors().col(d - 1 - j);
        }

        PlanSegments(eigenvalues.get(), bits_per_dimension, ComputeWidthKappas(executor), model);
        model.ComputeChunks();
        ComputeSegmentScales(model, seed);

        fit.rotation = skmeans::AllocateBuffer<float>(allocator, static_cast<size_t>(d) * d);
        Eigen::Map<eigen_matrix_t> rotation(fit.rotation.get(), d, d);
        std::unique_ptr<double[]> rotated_variances(new double[d]());
        for (uint32_t segment_idx = 0; segment_idx < model.num_segments; ++segment_idx) {
            const SAQSegment& segment = model.segments[segment_idx];
            const uint32_t length = segment.end - segment.start;
            auto segment_rotation_values =
                skmeans::AllocateBuffer<float>(allocator, static_cast<size_t>(length) * length);
            GenerateRotationMatrix(
                length,
                seed + segment_idx, // a different rotation per segment
                segment_rotation_values.get(),
                /*fix_signs=*/true
            );
            const Eigen::Map<const eigen_matrix_t> segment_rotation(
                segment_rotation_values.get(), length, length
            );
            rotation.middleCols(segment.start, length) =
                basis.middleCols(segment.start, length) * segment_rotation;
            // diag(P^T diag(eigenvalues) P): the variance of each rotated dimension
            for (uint32_t j = 0; j < length; ++j) {
                for (uint32_t i = 0; i < length; ++i) {
                    const double weight = segment_rotation(i, j);
                    rotated_variances[segment.start + j] +=
                        weight * weight * eigenvalues[segment.start + i];
                }
            }
        }

        model.rotated_mean.reset(new float[d]);
        Eigen::Map<Eigen::RowVectorXf>(model.rotated_mean.get(), d).noalias() =
            Eigen::Map<const Eigen::RowVectorXf>(mean.get(), d) * rotation;
        model.chunk_remaining_variances.reset(new float[model.num_chunks]);
        for (uint32_t chunk_idx = 0; chunk_idx < model.num_chunks; ++chunk_idx) {
            double remaining = 0.0;
            for (uint32_t j = (chunk_idx + 1) * SAQ_CHUNK_SIZE; j < d; ++j) {
                remaining += rotated_variances[j];
            }
            model.chunk_remaining_variances[chunk_idx] = static_cast<float>(remaining);
        }
        return fit;
    }

    void EncodeEmbedding(
        const float* PDX_RESTRICT embedding,
        const SAQModel& model,
        uint8_t* PDX_RESTRICT encoded,
        SAQEncodingScratch& scratch
    ) const {
        using computer = DistanceComputer<DistanceMetric::L2SQ, SAQ>;
        float* residual = scratch.residual.get();
        const float squared_norm = computer::ComputeResidual(
            embedding, model.rotated_mean.get(), model.num_dimensions, residual
        );
        uint8_t* levels = scratch.levels_by_chunk.get();
        uint8_t* factors = encoded + model.chunk_record_offsets[model.num_chunks];
        for (uint32_t segment_idx = 0; segment_idx < model.num_segments; ++segment_idx) {
            const SAQSegment& segment = model.segments[segment_idx];
            const float rescale = computer::EncodeSegment(
                residual + segment.start,
                segment.end - segment.start,
                segment.width,
                segment.scale,
                levels + segment.start
            );
            WriteHalf(rescale, factors + segment_idx * sizeof(uint16_t));
        }
        WriteHalf(squared_norm, factors + model.num_segments * sizeof(uint16_t));
        for (uint32_t chunk_idx = 0; chunk_idx < model.num_chunks; ++chunk_idx) {
            computer::PackRecord(
                levels + static_cast<size_t>(chunk_idx) * SAQ_CHUNK_SIZE,
                model.chunk_widths[chunk_idx],
                encoded + model.chunk_record_offsets[chunk_idx]
            );
        }
    }

    void DecodeEmbedding(
        const uint8_t* PDX_RESTRICT encoded,
        const SAQModel& model,
        float* PDX_RESTRICT out
    ) const {
        using computer = DistanceComputer<DistanceMetric::L2SQ, SAQ>;
        const uint8_t* factors = encoded + model.chunk_record_offsets[model.num_chunks];
        for (uint32_t chunk_idx = 0; chunk_idx < model.num_chunks; ++chunk_idx) {
            const uint32_t first_dimension = chunk_idx * SAQ_CHUNK_SIZE;
            computer::DecodeRecord(
                encoded + model.chunk_record_offsets[chunk_idx],
                model.chunk_widths[chunk_idx],
                ReadHalf(factors + model.chunk_segments[chunk_idx] * sizeof(uint16_t)),
                model.chunk_code_centers[chunk_idx],
                model.rotated_mean.get() + first_dimension,
                std::min(SAQ_CHUNK_SIZE, model.num_dimensions - first_dimension),
                out + first_dimension
            );
        }
    }

  private:
    // RaBitQ-Library's kTightStart and kNEnum (quantization/rabitq_impl.hpp): the scale search
    // starts at this fraction of its end (by ex_bits) and ends this many levels past the largest
    static constexpr std::array<double, SAQ_MAX_CODE_WIDTH + 1> TIGHT_START =
        {0.0, 0.15, 0.20, 0.52, 0.59, 0.71, 0.75, 0.77, 0.81};
    static constexpr double SCALE_SEARCH_EXTRA_LEVELS = 10.0;
    static constexpr size_t KAPPA_NUM_VECTORS = 3000;
    static constexpr size_t KAPPA_NUM_DIMENSIONS = 128;
    static constexpr double FACTOR_BITS = 16;
    static constexpr double PLAN_BIT_UNIT = 8;
    static constexpr size_t COVARIANCE_BLOCK_ROWS = 256;

    static double GetTightStart(uint32_t width) {
        return width <= 4 ? 0.0 : TIGHT_START[width - 1];
    }

    static size_t GetSampleRowIndex(size_t sample_idx, size_t num_samples, size_t num_embeddings) {
        return sample_idx * num_embeddings / num_samples;
    }

    // The scale t whose levels min(floor(t * |o|), max) maximize cos(level + 0.5, |o|), searched
    // over the scales where a level steps up (extended RaBitQ)
    static double FindBestScale(
        const double* absolute_values,
        size_t num_values,
        uint32_t ex_bits,
        double tight_start,
        int32_t* levels,
        std::pair<double, uint32_t>* heap
    ) {
        double max_value = 0.0;
        for (size_t i = 0; i < num_values; ++i) {
            max_value = std::max(max_value, absolute_values[i]);
        }
        if (max_value <= 0.0) {
            return 0.0;
        }
        const int32_t max_level = (int32_t{1} << ex_bits) - 1;
        const double t_end =
            (static_cast<double>(max_level) + SCALE_SEARCH_EXTRA_LEVELS) / max_value;
        const double t_start = t_end * tight_start;
        double squared_denominator = static_cast<double>(num_values) * 0.25;
        double numerator = 0.0;
        for (size_t i = 0; i < num_values; ++i) {
            int32_t level =
                static_cast<int32_t>(t_start * absolute_values[i] + SAQ_LEVEL_ROUNDING_EPSILON);
            level = std::min(level, max_level);
            levels[i] = level;
            squared_denominator += static_cast<double>(level * level + level);
            numerator += (static_cast<double>(level) + 0.5) * absolute_values[i];
        }
        double best_inner_product = numerator / std::sqrt(squared_denominator);
        double best_scale = t_start;
        size_t heap_size = 0;
        const auto greater = [](const std::pair<double, uint32_t>& a,
                                const std::pair<double, uint32_t>& b) { return a.first > b.first; };
        for (size_t i = 0; i < num_values; ++i) {
            if (levels[i] < max_level && absolute_values[i] > 0.0) {
                const double next_scale = static_cast<double>(levels[i] + 1) / absolute_values[i];
                if (next_scale < t_end) {
                    heap[heap_size++] = {next_scale, static_cast<uint32_t>(i)};
                }
            }
        }
        std::make_heap(heap, heap + heap_size, greater);
        while (heap_size > 0) {
            std::pop_heap(heap, heap + heap_size, greater);
            --heap_size;
            const auto [scale, i] = heap[heap_size];
            levels[i] += 1;
            const int32_t level = levels[i];
            squared_denominator += static_cast<double>(2 * level);
            numerator += absolute_values[i];
            const double inner_product = numerator / std::sqrt(squared_denominator);
            if (inner_product > best_inner_product) {
                best_inner_product = inner_product;
                best_scale = scale;
            }
            if (level < max_level) {
                const double next_scale = static_cast<double>(level + 1) / absolute_values[i];
                if (next_scale < t_end) {
                    heap[heap_size++] = {next_scale, i};
                    std::push_heap(heap, heap + heap_size, greater);
                }
            }
        }
        return best_scale;
    }

    // Each segment's scale: the mean best scale of SAQ_CONSTANT_SCALE_VECTORS random Gaussian
    // vectors of its length (RaBitQ-Library's get_const_scaling_factors). Each segment's random
    // rotation makes its coordinates close to Gaussian, so the scale needs no data
    static void ComputeSegmentScales(SAQModel& model, uint32_t seed) {
        uint32_t max_length = 0;
        for (uint32_t segment_idx = 0; segment_idx < model.num_segments; ++segment_idx) {
            max_length = std::max(
                max_length, model.segments[segment_idx].end - model.segments[segment_idx].start
            );
        }
        std::unique_ptr<double[]> values(new double[max_length]);
        std::unique_ptr<int32_t[]> levels(new int32_t[max_length]);
        std::unique_ptr<std::pair<double, uint32_t>[]> heap(
            new std::pair<double, uint32_t>[max_length]
        );
        std::mt19937 rng(seed);
        std::normal_distribution<double> gaussian(0.0, 1.0);
        for (uint32_t segment_idx = 0; segment_idx < model.num_segments; ++segment_idx) {
            SAQSegment& segment = model.segments[segment_idx];
            const uint32_t length = segment.end - segment.start;
            // 1-bit codes are the signs alone
            if (segment.width == 1) {
                segment.scale = 0.0f;
                continue;
            }
            double sum = 0.0;
            for (size_t vector_idx = 0; vector_idx < SAQ_CONSTANT_SCALE_VECTORS; ++vector_idx) {
                double squared_norm = 0.0;
                for (uint32_t j = 0; j < length; ++j) {
                    values[j] = gaussian(rng);
                    squared_norm += values[j] * values[j];
                }
                const double norm = std::sqrt(squared_norm);
                for (uint32_t j = 0; j < length; ++j) {
                    values[j] = std::abs(values[j]) / norm;
                }
                sum += FindBestScale(
                    values.get(),
                    length,
                    segment.width - 1,
                    GetTightStart(segment.width),
                    levels.get(),
                    heap.get()
                );
            }
            segment.scale =
                static_cast<float>(sum / static_cast<double>(SAQ_CONSTANT_SCALE_VECTORS));
        }
    }

    // The cosine between values and their width-bit code (levels + 0.5) under the exact scale
    // search
    static double ExactCodeCosine(
        const float* values,
        size_t num_values,
        uint32_t width,
        double* absolute_values,
        int32_t* levels,
        std::pair<double, uint32_t>* heap
    ) {
        const uint32_t ex_bits = width - 1;
        const int32_t max_level = (int32_t{1} << ex_bits) - 1;
        double squared_norm = 0.0;
        for (size_t j = 0; j < num_values; ++j) {
            squared_norm += static_cast<double>(values[j]) * values[j];
        }
        if (squared_norm <= 0.0) {
            return 1.0;
        }
        const double norm = std::sqrt(squared_norm);
        for (size_t j = 0; j < num_values; ++j) {
            absolute_values[j] = std::abs(static_cast<double>(values[j])) / norm;
        }
        const double scale =
            ex_bits > 0 ? FindBestScale(absolute_values, num_values, ex_bits, 0.0, levels, heap)
                        : 0.0;
        double inner_product = 0.0;
        double code_squared_norm = 0.0;
        for (size_t j = 0; j < num_values; ++j) {
            const int32_t level = std::min(
                static_cast<int32_t>(scale * absolute_values[j] + SAQ_LEVEL_ROUNDING_EPSILON),
                max_level
            );
            const double value = static_cast<double>(level) + 0.5;
            inner_product += value * absolute_values[j];
            code_squared_norm += value * value;
        }
        return inner_product / std::sqrt(code_squared_norm);
    }

    // Each worker owns a range of the covariance's columns and adds every centered block of sampled
    // rows to them, so its scratch is one block instead of a d x d sum
    static skmeans::AllocatedBuffer<float> ComputeCovariance(
        const float* PDX_RESTRICT embeddings,
        size_t num_embeddings,
        size_t num_samples,
        uint32_t num_dimensions,
        const float* mean,
        ParallelExecutor& executor,
        MemoryAllocator& allocator
    ) {
        const uint32_t d = num_dimensions;
        const size_t block_rows = std::min(COVARIANCE_BLOCK_ROWS, num_samples);
        auto covariance =
            skmeans::AllocateZeroedBuffer<float>(allocator, static_cast<size_t>(d) * d);
        auto centered_blocks =
            skmeans::AllocateBuffer<float>(allocator, executor.NumWorkers() * block_rows * d);
        const float inverse_num_samples = 1.0f / static_cast<float>(num_samples);
        executor.ParallelFor(d, [&](size_t column_begin, size_t column_end, size_t worker) {
            float* centered = centered_blocks.get() + worker * block_rows * d;
            for (size_t first_sample = 0; first_sample < num_samples; first_sample += block_rows) {
                const size_t rows = std::min(block_rows, num_samples - first_sample);
                for (size_t row_idx = 0; row_idx < rows; ++row_idx) {
                    const float* row =
                        embeddings +
                        GetSampleRowIndex(first_sample + row_idx, num_samples, num_embeddings) * d;
                    PDX_VECTORIZE_LOOP
                    for (uint32_t j = 0; j < d; ++j) {
                        centered[row_idx * d + j] = row[j] - mean[j];
                    }
                }
                skmeans::Sgemm(
                    'N',
                    'T',
                    static_cast<int>(d),
                    static_cast<int>(column_end - column_begin),
                    static_cast<int>(rows),
                    inverse_num_samples,
                    centered,
                    static_cast<int>(d),
                    centered + column_begin,
                    static_cast<int>(d),
                    1.0f,
                    covariance.get() + column_begin * d,
                    static_cast<int>(d)
                );
            }
        });
        return covariance;
    }

    static std::array<double, SAQ_MAX_CODE_WIDTH + 1> ComputeWidthKappas(ParallelExecutor& executor
    ) {
        std::unique_ptr<float[]> gaussian(new float[KAPPA_NUM_VECTORS * KAPPA_NUM_DIMENSIONS]);
        std::mt19937_64 rng(1);
        std::normal_distribution<float> normal;
        for (size_t i = 0; i < KAPPA_NUM_VECTORS * KAPPA_NUM_DIMENSIONS; ++i) {
            gaussian[i] = normal(rng);
        }
        std::array<double, SAQ_MAX_CODE_WIDTH + 1> kappas;
        kappas.fill(std::numeric_limits<double>::infinity());
        std::unique_ptr<double[]> cosines(new double[KAPPA_NUM_VECTORS]);
        for (uint32_t width = 1; width <= SAQ_MAX_CODE_WIDTH; ++width) {
            executor.ParallelFor(KAPPA_NUM_VECTORS, [&](size_t begin, size_t end, size_t) {
                std::unique_ptr<double[]> absolute_values(new double[KAPPA_NUM_DIMENSIONS]);
                std::unique_ptr<int32_t[]> levels(new int32_t[KAPPA_NUM_DIMENSIONS]);
                std::unique_ptr<std::pair<double, uint32_t>[]> heap(
                    new std::pair<double, uint32_t>[KAPPA_NUM_DIMENSIONS]
                );
                for (size_t i = begin; i < end; ++i) {
                    cosines[i] = ExactCodeCosine(
                        gaussian.get() + i * KAPPA_NUM_DIMENSIONS,
                        KAPPA_NUM_DIMENSIONS,
                        width,
                        absolute_values.get(),
                        levels.get(),
                        heap.get()
                    );
                }
            });
            double cosine_sum = 0.0;
            for (size_t i = 0; i < KAPPA_NUM_VECTORS; ++i) {
                cosine_sum += cosines[i];
            }
            const double cosine = cosine_sum / KAPPA_NUM_VECTORS;
            // The relative squared error of a width-bit code, from its mean cosine to Gaussian data
            kappas[width] = (1 - cosine * cosine) / (cosine * cosine);
        }
        return kappas;
    }

    static void PlanSegments(
        const double* eigenvalues,
        uint32_t bits_per_dimension,
        const std::array<double, SAQ_MAX_CODE_WIDTH + 1>& kappas,
        SAQModel& model
    ) {
        const uint32_t d = model.num_dimensions;
        if (d < SAQ_DP_PLAN_MIN_DIMENSIONS) {
            model.num_segments = 1;
            model.segments.reset(new SAQSegment[1]);
            model.segments[0] = {0, d, bits_per_dimension, 0.0f};
            return;
        }
        const size_t num_cuts = (d + SAQ_CHUNK_SIZE - 1) / SAQ_CHUNK_SIZE + 1;
        std::unique_ptr<uint32_t[]> cuts(new uint32_t[num_cuts]);
        for (size_t cut_idx = 0; cut_idx + 1 < num_cuts; ++cut_idx) {
            cuts[cut_idx] = static_cast<uint32_t>(cut_idx * SAQ_CHUNK_SIZE);
        }
        cuts[num_cuts - 1] = d;
        std::unique_ptr<double[]> cumulative_energy(new double[d + 1]);
        cumulative_energy[0] = 0.0;
        for (uint32_t i = 0; i < d; ++i) {
            cumulative_energy[i + 1] = cumulative_energy[i] + eigenvalues[i];
        }
        const int max_units =
            static_cast<int>(static_cast<double>(bits_per_dimension) * d - FACTOR_BITS) /
            static_cast<int>(PLAN_BIT_UNIT);
        const size_t table_size = num_cuts * (max_units + 1);
        const double infinity = std::numeric_limits<double>::infinity();
        std::unique_ptr<double[]> best(new double[table_size]);
        std::unique_ptr<int[]> previous_cut(new int[table_size]);
        std::unique_ptr<int[]> previous_width(new int[table_size]);
        std::unique_ptr<int[]> previous_units(new int[table_size]);
        const auto at = [max_units](size_t cut_idx, int units) {
            return cut_idx * (max_units + 1) + units;
        };
        // Fills the table; returns the least distortion of a whole plan and its units. With
        // use_pdx_u8_head, the head chunk is its own 8-bit segment in the PDX U8 layout, which
        // every vector's scan reads unpacked
        const auto find_plan = [&](bool use_pdx_u8_head) {
            std::fill_n(best.get(), table_size, infinity);
            std::fill_n(previous_cut.get(), table_size, -1);
            std::fill_n(previous_width.get(), table_size, -1);
            std::fill_n(previous_units.get(), table_size, -1);
            best[at(0, 0)] = 0;
            for (size_t j = 1; j < num_cuts; ++j) {
                for (size_t i = 0; i < j; ++i) {
                    if (use_pdx_u8_head && i == 0 && j != 1) {
                        continue;
                    }
                    const int min_width =
                        use_pdx_u8_head && i == 0 ? static_cast<int>(SAQ_MAX_CODE_WIDTH) : 1;
                    const double length = cuts[j] - cuts[i];
                    // A partial last chunk's record still stores all of its 64 dimensions
                    const double stored_length =
                        AlignValue<uint32_t, SAQ_CHUNK_SIZE>(cuts[j]) - cuts[i];
                    const double energy = cumulative_energy[cuts[j]] - cumulative_energy[cuts[i]];
                    for (int width = min_width; width <= static_cast<int>(SAQ_MAX_CODE_WIDTH);
                         ++width) {
                        // Distortion of the segment at this width, kappa * energy^2 / length; it
                        // costs its codes plus a 16-bit rescale, in units of PLAN_BIT_UNIT bits
                        const double cost = kappas[width] * energy * energy / length;
                        const int units = static_cast<int>(
                            std::ceil((stored_length * width + FACTOR_BITS) / PLAN_BIT_UNIT)
                        );
                        for (int u = 0; u + units <= max_units; ++u) {
                            const double previous_cost = best[at(i, u)];
                            if (previous_cost == infinity) {
                                continue;
                            }
                            const double total_cost = previous_cost + cost;
                            if (total_cost < best[at(j, u + units)]) {
                                best[at(j, u + units)] = total_cost;
                                previous_cut[at(j, u + units)] = static_cast<int>(i);
                                previous_width[at(j, u + units)] = width;
                                previous_units[at(j, u + units)] = u;
                            }
                        }
                    }
                }
            }
            std::pair<double, int> best_plan{infinity, 0};
            for (int u = 0; u <= max_units; ++u) {
                if (best[at(num_cuts - 1, u)] < best_plan.first) {
                    best_plan = {best[at(num_cuts - 1, u)], u};
                }
            }
            return best_plan;
        };
        auto plan = find_plan(/*use_pdx_u8_head=*/d >= SAQ_PDX_U8_HEAD_MIN_DIMENSIONS);
        if (plan.first == infinity) {
            plan = find_plan(/*use_pdx_u8_head=*/false);
        }
        const auto [best_cost, best_units] = plan;
        if (best_cost == infinity) {
            throw std::invalid_argument(
                "No SAQ plan of " + std::to_string(d) + " dimensions fits " +
                std::to_string(bits_per_dimension) + " bits per dimension: use more bits"
            );
        }
        uint32_t num_segments = 0;
        for (int cut = static_cast<int>(num_cuts) - 1, units = best_units; cut > 0;
             ++num_segments) {
            const size_t cell = at(static_cast<size_t>(cut), units);
            cut = previous_cut[cell];
            units = previous_units[cell];
        }
        model.num_segments = num_segments;
        model.segments.reset(new SAQSegment[num_segments]);
        int cut = static_cast<int>(num_cuts) - 1;
        int units = best_units;
        for (uint32_t segment_idx = num_segments; segment_idx-- > 0;) {
            const size_t cell = at(static_cast<size_t>(cut), units);
            model.segments[segment_idx] = {
                cuts[previous_cut[cell]],
                cuts[cut],
                static_cast<uint32_t>(previous_width[cell]),
                0.0f
            };
            cut = previous_cut[cell];
            units = previous_units[cell];
        }
    }
};

class SAQQuery {
  public:
    static constexpr float MAX_ABSOLUTE_CODE = 127.0f;

    // transformed_query: normalized if the index needs it, then rotated
    SAQQuery(const float* PDX_RESTRICT transformed_query, const SAQModel& model)
        : num_chunks(model.num_chunks),
          codes(new int8_t[static_cast<size_t>(model.num_chunks) * SAQ_CHUNK_SIZE]()),
          chunk_dot_scales(new float[model.num_chunks]),
          chunk_dot_offsets(new float[model.num_chunks]),
          chunk_squared_norms(new float[model.num_chunks]) {
        using computer = DistanceComputer<DistanceMetric::L2SQ, F32>;
        float residual[SAQ_CHUNK_SIZE];
        for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            const uint32_t first_dimension = chunk_idx * SAQ_CHUNK_SIZE;
            const uint32_t chunk_dimensions =
                std::min(SAQ_CHUNK_SIZE, model.num_dimensions - first_dimension);
            float chunk_squared_norm = 0.0f;
            for (uint32_t j = 0; j < chunk_dimensions; ++j) {
                residual[j] = transformed_query[first_dimension + j] -
                              model.rotated_mean[first_dimension + j];
                chunk_squared_norm += residual[j] * residual[j];
            }
            chunk_squared_norms[chunk_idx] = chunk_squared_norm;
            residual_squared_norm += chunk_squared_norm;
            float lower, upper;
            computer::MinMax(residual, chunk_dimensions, lower, upper);
            const float max_absolute_value = std::max(-lower, upper);
            const float scale =
                max_absolute_value > 0 ? max_absolute_value / MAX_ABSOLUTE_CODE : 1.0f;
            int8_t* chunk_codes = codes.get() + first_dimension;
            computer::QuantizeToInt8(residual, chunk_dimensions, 1.0f / scale, chunk_codes);
            int32_t codes_sum = 0;
            for (uint32_t j = 0; j < chunk_dimensions; ++j) {
                codes_sum += chunk_codes[j];
            }
            // The kernels turn <codes, embedding levels> into -2 <query, decoded embedding>
            chunk_dot_scales[chunk_idx] = -2.0f * scale;
            chunk_dot_offsets[chunk_idx] = chunk_dot_scales[chunk_idx] *
                                           model.chunk_code_centers[chunk_idx] *
                                           static_cast<float>(codes_sum);
        }
    }

    const int8_t* GetCodesPtr() const { return codes.get(); }
    const float* GetChunkDotScalesPtr() const { return chunk_dot_scales.get(); }
    const float* GetChunkDotOffsetsPtr() const { return chunk_dot_offsets.get(); }
    const float* GetChunkSquaredNormsPtr() const { return chunk_squared_norms.get(); }
    float GetResidualSquaredNorm() const { return residual_squared_norm; }

  private:
    uint32_t num_chunks;
    std::unique_ptr<int8_t[]> codes;
    std::unique_ptr<float[]> chunk_dot_scales;
    std::unique_ptr<float[]> chunk_dot_offsets;
    std::unique_ptr<float[]> chunk_squared_norms;
    float residual_squared_norm = 0.0f;
};

// Every vector's estimate starts at |r|^2 + |q_c|^2, then each chunk adds its -2 <q_c, r> term
inline void StartSAQEstimates(
    const uint16_t* squared_norms,
    size_t n_vectors,
    float query_squared_norm,
    float* estimates
) {
    nk_cast(squared_norms, nk_f16_k, n_vectors, estimates, nk_f32_k);
    PDX_VECTORIZE_LOOP
    for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
        estimates[vector_idx] += query_squared_norm;
    }
}

// StartSAQEstimates for the vectors at positions
inline void StartSAQEstimatesPruning(
    const uint16_t* squared_norms,
    const uint32_t* positions,
    size_t n_positions,
    float query_squared_norm,
    float* estimates
) {
    for (size_t position_idx = 0; position_idx < n_positions; ++position_idx) {
        estimates[position_idx] =
            ReadHalf(reinterpret_cast<const uint8_t*>(squared_norms + positions[position_idx])) +
            query_squared_norm;
    }
}

// Every vector's estimate over all of its chunks, from a cluster's SAQ data whose sections hold
// stride slots; estimates has room for n_vectors + SAQ_SCAN_PADDING values
inline void ComputeSAQEstimates(
    const uint8_t* data,
    size_t n_vectors,
    size_t stride,
    const SAQModel& model,
    const SAQQuery& query,
    float* estimates
) {
    using computer = DistanceComputer<DistanceMetric::L2SQ, SAQ>;
    const auto* rescales = reinterpret_cast<const uint16_t*>(
        data + stride * model.chunk_record_offsets[model.num_chunks]
    );
    const int8_t* query_codes = query.GetCodesPtr();
    const float* dot_scales = query.GetChunkDotScalesPtr();
    const float* dot_offsets = query.GetChunkDotOffsetsPtr();
    const uint16_t* squared_norms = rescales + model.num_segments * stride;
    const uint32_t first_record_chunk = model.HasPDXU8Head() ? 1 : 0;
    if (model.HasPDXU8Head()) {
        computer::ComputeHeadEstimates(
            data,
            n_vectors,
            stride,
            query_codes,
            dot_scales[0],
            dot_offsets[0],
            query.GetResidualSquaredNorm(),
            rescales,
            squared_norms,
            estimates
        );
    } else {
        StartSAQEstimates(squared_norms, n_vectors, query.GetResidualSquaredNorm(), estimates);
    }
    for (uint32_t chunk_idx = first_record_chunk; chunk_idx < model.num_chunks; ++chunk_idx) {
        computer::GetAccumulateTailEstimates(model.chunk_widths[chunk_idx])(
            data + stride * model.chunk_record_offsets[chunk_idx],
            n_vectors,
            query_codes + static_cast<size_t>(chunk_idx) * SAQ_CHUNK_SIZE,
            dot_scales[chunk_idx],
            dot_offsets[chunk_idx],
            rescales + model.chunk_segments[chunk_idx] * stride,
            estimates
        );
    }
}

} // namespace PDX
