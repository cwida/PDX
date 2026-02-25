#pragma once

#include "pdx/common.hpp"
#include "pdx/distance_computers/base_computers.hpp"
#include <Eigen/Dense>
#include <omp.h>
#include <queue>
#include <random>

#ifdef HAS_FFTW
#include <fftw3.h>
#endif

namespace PDX {

class ADSamplingPruner {
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
            fftwf_init_threads();
            matrix.resize(1, num_dimensions);
            std::uniform_int_distribution<int> dist(0, 1);
            for (size_t i = 0; i < num_dimensions; ++i) {
                matrix(i) = dist(gen) ? 1.0f : -1.0f;
            }
            BuildFlipMasks();
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
            fftwf_init_threads();
            matrix = Eigen::Map<const matrix_t>(matrix_p, 1, num_dimensions);
            BuildFlipMasks();
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

    float GetPruningThreshold(
        uint32_t,
        std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator>& heap,
        const uint32_t current_dimension_idx
    ) const {
        float ratio = current_dimension_idx == num_dimensions ? 1 : ratios[current_dimension_idx];
        return heap.top().distance * ratio;
    }

    void PreprocessQuery(
        const float* PDX_RESTRICT const raw_query_embedding,
        float* PDX_RESTRICT const output_query_embedding
    ) const {
        PreprocessEmbeddings(raw_query_embedding, output_query_embedding, 1);
    }

    void PreprocessEmbeddings(
        const float* PDX_RESTRICT const input_embeddings,
        float* PDX_RESTRICT const output_embeddings,
        const size_t num_embeddings
    ) const {
        Rotate(input_embeddings, output_embeddings, num_embeddings);
    }

  private:
    float pruning_aggressiveness = ADSAMPLING_PRUNING_AGGRESIVENESS;
    matrix_t matrix;
    std::vector<float> ratios;
    std::vector<uint32_t> flip_masks;

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
        return static_cast<float>(visited_dimensions) / num_dimensions *
               (1.0 + pruning_aggressiveness / std::sqrt(visited_dimensions)) *
               (1.0 + pruning_aggressiveness / std::sqrt(visited_dimensions));
    }

    void BuildFlipMasks() {
        flip_masks.resize(num_dimensions);
        for (size_t i = 0; i < num_dimensions; ++i) {
            flip_masks[i] = (matrix(i) < 0.0f ? 0x80000000u : 0u);
        }
    }

    void FlipSign(const float* data, float* out, const size_t n) const {
#pragma omp parallel for num_threads(PDX::g_n_threads)
        for (size_t i = 0; i < n; ++i) {
            const size_t offset = i * num_dimensions;
            flip_sign_fn::FlipSign(data + offset, out + offset, flip_masks.data(), num_dimensions);
        }
    }

    void Rotate(
        const float* PDX_RESTRICT const embeddings,
        float* PDX_RESTRICT const out_buffer,
        const size_t n
    ) const {
#ifdef HAS_FFTW
        if (UsesDCTRotation()) {
            Eigen::Map<matrix_t> out(out_buffer, n, num_dimensions);
            FlipSign(embeddings, out_buffer, n);
            const float s0 = std::sqrt(1.0f / (4.0f * num_dimensions));
            const float s = std::sqrt(1.0f / (2.0f * num_dimensions));
            if (n == 1) {
                fftwf_plan plan = fftwf_plan_r2r_1d(
                    num_dimensions, out.data(), out.data(), FFTW_REDFT10, FFTW_ESTIMATE
                );
                fftwf_execute(plan);
                fftwf_destroy_plan(plan);
            } else {
                int n0 = static_cast<int>(num_dimensions);
                int howmany = static_cast<int>(n);
                fftw_r2r_kind kind[1] = {FFTW_REDFT10};
                auto flag = FFTW_MEASURE;
                if (IsPowerOf2(num_dimensions)) {
                    flag = FFTW_ESTIMATE;
                }
                fftwf_plan_with_nthreads(static_cast<int>(PDX::g_n_threads));
                fftwf_plan plan = fftwf_plan_many_r2r(
                    1, &n0, howmany, out.data(), NULL, 1, n0, out.data(), NULL, 1, n0, kind, flag
                );
                fftwf_execute(plan);
                fftwf_destroy_plan(plan);
            }
            out.col(0) *= s0;
            out.rightCols(num_dimensions - 1) *= s;
            return;
        }
#endif
        const char trans_a = 'N';
        const char trans_b = 'N';
        const float alpha = 1.0f;
        const float beta = 0.0f;
        int dim = static_cast<int>(num_dimensions);
        int n_blas = static_cast<int>(n);
        sgemm_(
            &trans_a,
            &trans_b,
            &dim,
            &n_blas,
            &dim,
            &alpha,
            matrix.data(),
            &dim,
            embeddings,
            &dim,
            &beta,
            out_buffer,
            &dim
        );
    }
};

} // namespace PDX
