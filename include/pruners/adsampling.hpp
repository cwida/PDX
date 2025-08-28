#ifndef PDX_ADSAMPLING_U8_HPP
#define PDX_ADSAMPLING_U8_HPP

#include <Eigen/Eigen/Dense>
#include <queue>
#include <utility>
#ifdef HAS_FFTW
    #include <fftw3.h>
#endif

namespace PDX {

static std::vector<float> ratios{};

/******************************************************************
 * ADSampling pruner
 ******************************************************************/
template<Quantization q=F32>
class ADSamplingPruner {
    using DISTANCES_TYPE = DistanceType_t<q>;
    using KNNCandidate_t = KNNCandidate<q>;
    using VectorComparator_t = VectorComparator<q>;

public:

    uint32_t num_dimensions;

    ADSamplingPruner(uint32_t num_dimensions, float epsilon0, float * matrix_p)
            : num_dimensions(num_dimensions), epsilon0(epsilon0){
        ratios.resize(num_dimensions);
        for (size_t i = 0; i < num_dimensions; ++i) {
            ratios[i] = GetRatio(i);
        }
#ifdef HAS_FFTW
        if (num_dimensions >= D_THRESHOLD_FOR_DCT_ROTATION) {
            matrix = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrix_p, 1, num_dimensions);
        } else {
            matrix = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrix_p, num_dimensions, num_dimensions);
        }
#else
        matrix = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrix_p, num_dimensions, num_dimensions);
#endif
    }

    void SetEpsilon0(float epsilon0) {
        ADSamplingPruner::epsilon0 = epsilon0;
        for (size_t i = 0; i < num_dimensions; ++i) {
            ratios[i] = GetRatio(i);
        }
    }

    void SetMatrix(const Eigen::MatrixXf &matrix) {
        ADSamplingPruner::matrix = matrix;
    }

    template<Quantization Q=q>
    DistanceType_t<Q> GetPruningThreshold(
        uint32_t k,
        std::priority_queue<KNNCandidate<Q>, std::vector<KNNCandidate<Q>>, VectorComparator<Q>> &heap,
        const uint32_t current_dimension_idx
    ) {
        float ratio = current_dimension_idx == num_dimensions ? 1 : ratios[current_dimension_idx];
        //return std::numeric_limits<DistanceType_t<Q>>::max();
        return heap.top().distance * ratio;
    }

    void PreprocessQuery(float *raw_query, float * query) {
        Multiply(raw_query, query, num_dimensions);
    }

private:
    float epsilon0 = 2.1;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix;

    float GetRatio(const size_t &visited_dimensions) {
        if (visited_dimensions == 0) {
            return 1;
        }
        if (visited_dimensions == (int) num_dimensions) {
            return 1.0;
        }
        return 1.0 * visited_dimensions / ((int) num_dimensions) *
               (1.0 + epsilon0 / std::sqrt(visited_dimensions)) * (1.0 + epsilon0 / std::sqrt(visited_dimensions));
    }

    void Multiply(float *raw_query, float *query, uint32_t num_dimensions) {
        Eigen::Map<const Eigen::RowVectorXf> query_matrix(raw_query, num_dimensions);
        Eigen::Map<Eigen::RowVectorXf> output(query, num_dimensions);
#ifdef HAS_FFTW
        if (num_dimensions >= D_THRESHOLD_FOR_DCT_ROTATION) {
            Eigen::RowVectorXf first_row = matrix.row(0);
            Eigen::RowVectorXf pre_output = query_matrix.array() * first_row.array();
            fftwf_plan plan = fftwf_plan_r2r_1d(num_dimensions, pre_output.data(), output.data(), FFTW_REDFT10, FFTW_ESTIMATE);
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
            output[0] *= std::sqrt(1.0 / (4 * num_dimensions));
            for (int i = 1; i < num_dimensions; ++i)
                output[i] *= std::sqrt(1.0 / (2 * num_dimensions));
            return;
        }
#endif
        output.noalias() = query_matrix * matrix;
    }

};

} // namespace PDX

#endif // PDX_ADSAMPLING_U8_HPP
