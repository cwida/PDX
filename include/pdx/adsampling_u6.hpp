#ifndef EMBEDDINGSEARCH_ADSAMPLING_U6_HPP
#define EMBEDDINGSEARCH_ADSAMPLING_U6_HPP

#include "pdx/pdxearch_u6x8.hpp"
#include <Eigen/Eigen/Dense>
#include <utility>

namespace PDX {

static std::vector<float> ratios{};

/******************************************************************
 * ADSampling + PDXearch
 * Overrides GetPruningThreshold() from pdxearch.hpp to use the ADSampling threshold
 * Overrides PreprocessQuery() to use ADSampling query preprocessing
 ******************************************************************/
class ADSamplingSearcherU6 : public PDXearchU6x8<L2> {
public:
    ADSamplingSearcherU6(IndexPDXIVFFlatU6x8 &pdx_index,
                       size_t ivf_nprobe, float epsilon0, Eigen::MatrixXf matrix,
                       PDXearchDimensionsOrder dimension_order)
            : PDXearchU6x8<L2>(pdx_index,
                           ivf_nprobe,
                           1,
                           dimension_order),
              epsilon0(epsilon0), matrix(std::move(matrix)) {
        // Initialize ratios for the dataset
        ratios.resize(pdx_data.num_dimensions);
        for (size_t i = 0; i < pdx_data.num_dimensions; ++i) {
            ratios[i] = GetRatio(i);
        }
    }

    void SetEpsilon0(float epsilon0) {
        ADSamplingSearcherU6::epsilon0 = epsilon0;
    }

    void SetMatrix(const Eigen::MatrixXf &matrix) {
        ADSamplingSearcherU6::matrix = matrix;
    }

    inline void GetPruningThreshold(uint32_t k, std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator> &heap) override {
        float ratio = current_dimension_idx == pdx_data.num_dimensions ? 1 : ratios[current_dimension_idx];
        pruning_threshold = heap.top().distance * ratio;
    }

    void PreprocessQuery(float *raw_query, float * query) override {
        Multiply(raw_query, query, pdx_data.num_dimensions);
    }

private:
    float epsilon0 = 2.1;
    Eigen::MatrixXf matrix;

    inline float GetRatio(const size_t &visited_dimensions) {
        if(visited_dimensions == (int) pdx_data.num_dimensions) {
            return 1.0;
        }
        return 1.0 * visited_dimensions / ((int) pdx_data.num_dimensions) *
               (1.0 + epsilon0 / std::sqrt(visited_dimensions)) * (1.0 + epsilon0 / std::sqrt(visited_dimensions));
    }

    // Improved transformation (2x - 100x faster than the original one, depending on D)
    void Multiply(float *raw_query, float *query, uint32_t num_dimensions) {
        Eigen::MatrixXf query_matrix = Eigen::Map<Eigen::MatrixXf>(raw_query, 1, num_dimensions);
        Eigen::MatrixXf mul_result(1, num_dimensions);
        mul_result = query_matrix * matrix;
        for (size_t i = 0; i < num_dimensions; ++i){
            query[i] = mul_result(0, i);
        }
    }

};

} // namespace PDX

#endif //EMBEDDINGSEARCH_ADSAMPLING_HPP
