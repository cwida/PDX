#ifndef PDX_ADSAMPLING_U8_HPP
#define PDX_ADSAMPLING_U8_HPP

#include <Eigen/Eigen/Dense>
#include <queue>
#include <utility>

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

    ADSamplingPruner(uint32_t num_dimensions, float epsilon0, Eigen::MatrixXf matrix)
            : num_dimensions(num_dimensions), epsilon0(epsilon0), matrix(std::move(matrix)) {
        ratios.resize(num_dimensions);
        for (size_t i = 0; i < num_dimensions; ++i) {
            ratios[i] = GetRatio(i);
        }
    }

    void SetEpsilon0(float epsilon0) {
        ADSamplingPruner::epsilon0 = epsilon0;
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
        return heap.top().distance * ratio;
    }

    void PreprocessQuery(float *raw_query, float * query) {
        Multiply(raw_query, query, num_dimensions);
    }

private:
    float epsilon0 = 2.1;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix;

    float GetRatio(const size_t &visited_dimensions) {
        if(visited_dimensions == (int) num_dimensions) {
            return 1.0;
        }
        return 1.0 * visited_dimensions / ((int) num_dimensions) *
               (1.0 + epsilon0 / std::sqrt(visited_dimensions)) * (1.0 + epsilon0 / std::sqrt(visited_dimensions));
    }

    // Improved transformation (2x - 100x faster than the original one, depending on D)
    void Multiply(float *raw_query, float *query, uint32_t num_dimensions) {
        Eigen::Map<const Eigen::RowVectorXf> query_matrix(raw_query, num_dimensions);
        Eigen::Map<Eigen::RowVectorXf> output(query, num_dimensions);
        output.noalias() = query_matrix * matrix;
    }

};

} // namespace PDX

#endif //PDX_ADSAMPLING_HPP
