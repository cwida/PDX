#pragma once

#include "pdx/common.hpp"
#include "pdx/distance_computers/base_computers.hpp"
#include "superkmeans/common.h"
#include <Eigen/Dense>
#include <algorithm>
#include <queue>
#include <random>

#ifdef HAS_FFTW
#include <fftw3.h>
#endif

namespace PDX {

class ADSamplingPruner : public skmeans::ExecutorHolder {
    using matrix_t = eigen_matrix_t;
    using flip_sign_fn = DistanceComputer<DistanceMetric::L2SQ, F32>;

  public:
    const uint32_t num_dimensions;

    ADSamplingPruner(const uint32_t num_dimensions, const int32_t seed)
        : num_dimensions(num_dimensions) {
        ratios.resize(num_dimensions);
        for (size_t i = 0; i < num_dimensions; ++i) {
            ratios[i] = GetRatio(i);
        }
        std::mt19937 gen(seed);
        bool matrix_created = false;
#ifdef HAS_FFTW
        if (UsesDCTRotation()) {
            matrix.resize(1, num_dimensions);
            std::uniform_int_distribution<int> dist(0, 1);
            for (size_t i = 0; i < num_dimensions; ++i) {
                matrix(i) = dist(gen) ? 1.0f : -1.0f;
            }
            BuildFlipMasks();
            CacheSingleQueryPlan();
            matrix_created = true;
        }
#endif
        if (!matrix_created) {
            std::normal_distribution<float> normal_dist;
            Eigen::MatrixXf random_matrix = Eigen::MatrixXf::Zero(
                static_cast<Eigen::Index>(num_dimensions), static_cast<Eigen::Index>(num_dimensions)
            );
            for (size_t i = 0; i < num_dimensions; ++i) {
                for (size_t j = 0; j < num_dimensions; ++j) {
                    random_matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
                        normal_dist(gen);
                }
            }
            const Eigen::HouseholderQR<Eigen::MatrixXf> qr(random_matrix);
            matrix = qr.householderQ() * Eigen::MatrixXf::Identity(num_dimensions, num_dimensions);
        }
    }

    ADSamplingPruner(const uint32_t num_dimensions, const float* matrix_p)
        : num_dimensions(num_dimensions) {
        ratios.resize(num_dimensions);
        for (size_t i = 0; i < num_dimensions; ++i) {
            ratios[i] = GetRatio(i);
        }
#ifdef HAS_FFTW
        if (UsesDCTRotation()) {
            matrix = Eigen::Map<const matrix_t>(matrix_p, 1, num_dimensions);
            BuildFlipMasks();
            CacheSingleQueryPlan();
        } else {
            matrix = Eigen::Map<const matrix_t>(matrix_p, num_dimensions, num_dimensions);
        }
#else
        matrix = Eigen::Map<const matrix_t>(matrix_p, num_dimensions, num_dimensions);
#endif
    }

    void SetPruningAggresiveness(const float pruning_aggressiveness) {
        ADSamplingPruner::pruning_aggressiveness = pruning_aggressiveness;
        for (size_t i = 0; i < num_dimensions; ++i) {
            ratios[i] = GetRatio(i);
        }
    }

    void SetMatrix(const Eigen::MatrixXf& matrix) { ADSamplingPruner::matrix = matrix; }

    const matrix_t& GetMatrix() const { return matrix; }

    float GetPruningThreshold(uint32_t, Heap& heap, const uint32_t current_dimension_idx) const {
        float ratio = current_dimension_idx == num_dimensions ? 1 : ratios[current_dimension_idx];
        return heap.top().distance * ratio;
    }

    void PreprocessQuery(
        const float* PDX_RESTRICT const raw_query_embedding,
        float* PDX_RESTRICT const output_query_embedding
    ) const {
        PreprocessEmbeddings(raw_query_embedding, output_query_embedding, 1);
    }

    // Runs on the given executor, else on the bound one (serial when unbound).
    void PreprocessEmbeddings(
        const float* PDX_RESTRICT const input_embeddings,
        float* PDX_RESTRICT const output_embeddings,
        const size_t num_embeddings,
        ParallelExecutor* executor = nullptr
    ) const {
        Rotate(
            input_embeddings,
            output_embeddings,
            num_embeddings,
            executor != nullptr ? *executor : GetExecutor()
        );
    }

    ~ADSamplingPruner() {
#ifdef HAS_FFTW
        if (single_query_plan) {
            fftwf_destroy_plan(single_query_plan);
        }
#endif
    }

    ADSamplingPruner(const ADSamplingPruner&) = delete;
    ADSamplingPruner& operator=(const ADSamplingPruner&) = delete;

  private:
    float pruning_aggressiveness = ADSAMPLING_PRUNING_AGGRESIVENESS;
    matrix_t matrix;
    std::vector<float> ratios;
    std::vector<uint32_t> flip_masks;
#ifdef HAS_FFTW
    fftwf_plan single_query_plan = nullptr;
#endif

    bool UsesDCTRotation() const {
#ifdef HAS_FFTW
#ifdef __AVX2__
        return num_dimensions >= D_THRESHOLD_FOR_DCT_ROTATION && IsPowerOf2(num_dimensions);
#else
        return num_dimensions >= D_THRESHOLD_FOR_DCT_ROTATION;
#endif
#else
        return false;
#endif
    }

    float GetRatio(const size_t& visited_dimensions) const {
        if (visited_dimensions == 0) {
            return 1;
        }
        if (visited_dimensions == num_dimensions) {
            return 1.0;
        }
        return static_cast<float>(
            static_cast<float>(visited_dimensions) / num_dimensions *
            (1.0 + pruning_aggressiveness / std::sqrt(visited_dimensions)) *
            (1.0 + pruning_aggressiveness / std::sqrt(visited_dimensions))
        );
    }

    void BuildFlipMasks() {
        flip_masks.resize(num_dimensions);
        for (size_t i = 0; i < num_dimensions; ++i) {
            flip_masks[i] = (matrix(i) < 0.0f ? 0x80000000u : 0u);
        }
    }

#ifdef HAS_FFTW
    void CacheSingleQueryPlan() {
        std::unique_ptr<float[]> tmp(new float[num_dimensions]);
        single_query_plan =
            fftwf_plan_r2r_1d(num_dimensions, tmp.get(), tmp.get(), FFTW_REDFT10, FFTW_ESTIMATE);
    }
#endif

    void FlipSign(const float* data, float* out, const size_t n, ParallelExecutor& executor) const {
        executor.ParallelFor(n, [&](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                const size_t offset = i * num_dimensions;
                flip_sign_fn::FlipSign(
                    data + offset, out + offset, flip_masks.data(), num_dimensions
                );
            }
        });
    }

#ifdef HAS_FFTW
    // Plans on the calling thread (the FFTW planner is not thread-safe), executes blocks in
    // parallel.
    void ParallelDCT(float* out, const size_t n, ParallelExecutor& executor) const {
        if (n == 0) {
            return;
        }
        const int n0 = static_cast<int>(num_dimensions);
        fftw_r2r_kind kind = FFTW_REDFT10;
        const unsigned flag =
            (IsPowerOf2(num_dimensions) ? FFTW_ESTIMATE : FFTW_MEASURE) | FFTW_UNALIGNED;
        const size_t block_rows = std::min(skmeans::ROTATION_BLOCK_SIZE, n);
        const size_t tail_rows = n % block_rows;
        std::unique_ptr<float[]> scratch(new float[block_rows * num_dimensions]);
        auto make_plan = [&](size_t rows) {
            const int howmany = static_cast<int>(rows);
            return fftwf_plan_many_r2r(
                1,
                &n0,
                howmany,
                scratch.get(),
                nullptr,
                1,
                n0,
                scratch.get(),
                nullptr,
                1,
                n0,
                &kind,
                flag
            );
        };
        fftwf_plan block_plan = make_plan(block_rows);
        fftwf_plan tail_plan = tail_rows > 0 ? make_plan(tail_rows) : nullptr;
        const size_t n_blocks = (n + block_rows - 1) / block_rows;
        executor.ParallelFor(n_blocks, [&](size_t block_begin, size_t block_end, size_t) {
            for (size_t block = block_begin; block < block_end; ++block) {
                const size_t row = block * block_rows;
                float* rows_p = out + row * num_dimensions;
                fftwf_execute_r2r(row + block_rows <= n ? block_plan : tail_plan, rows_p, rows_p);
            }
        });
        fftwf_destroy_plan(block_plan);
        if (tail_plan != nullptr) {
            fftwf_destroy_plan(tail_plan);
        }
    }
#endif

    void Rotate(
        const float* PDX_RESTRICT const embeddings,
        float* PDX_RESTRICT const out_buffer,
        const size_t n,
        ParallelExecutor& executor
    ) const {
#ifdef HAS_FFTW
        if (UsesDCTRotation()) {
            Eigen::Map<matrix_t> out(out_buffer, n, num_dimensions);
            FlipSign(embeddings, out_buffer, n, executor);
            const float s0 = std::sqrt(1.0f / (4.0f * num_dimensions));
            const float s = std::sqrt(1.0f / (2.0f * num_dimensions));
            if (n == 1) {
                fftwf_execute_r2r(single_query_plan, out.data(), out.data());
            } else {
                ParallelDCT(out_buffer, n, executor);
            }
            out.col(0) *= s0;
            out.rightCols(num_dimensions - 1) *= s;
            return;
        }
#endif
        // Single-threaded GEMMs over blocks of ROTATION_BLOCK_SIZE rows, run in parallel
        const int dim = static_cast<int>(num_dimensions);
        const size_t n_blocks =
            (n + skmeans::ROTATION_BLOCK_SIZE - 1) / skmeans::ROTATION_BLOCK_SIZE;
        executor.ParallelFor(n_blocks, [&](size_t block_begin, size_t block_end, size_t) {
            for (size_t block = block_begin; block < block_end; ++block) {
                const size_t row = block * skmeans::ROTATION_BLOCK_SIZE;
                const int n_rows =
                    static_cast<int>(std::min(skmeans::ROTATION_BLOCK_SIZE, n - row));
                skmeans::Sgemm(
                    'N',
                    'N',
                    dim,
                    n_rows,
                    dim,
                    1.0f,
                    matrix.data(),
                    dim,
                    embeddings + row * num_dimensions,
                    dim,
                    0.0f,
                    out_buffer + row * num_dimensions,
                    dim
                );
            }
        });
    }
};

} // namespace PDX
