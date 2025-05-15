#ifndef PDX_ADSAMPLING_U8_HPP
#define PDX_ADSAMPLING_U8_HPP

#include "pdx/pdxearch.hpp"
#include <Eigen/Eigen/Dense>
#include <utility>

namespace PDX {

static std::vector<float> ratios{};

/******************************************************************
 * ADSampling + PDXearch
 * Overrides GetPruningThreshold() from pdxearch.hpp to use the ADSampling threshold
 * Overrides PreprocessQuery() to use ADSampling query preprocessing
 ******************************************************************/
template<Quantization q=F32>
class ADSamplingSearcher : public PDXearch<q> {
    using DISTANCES_TYPE = DistanceType_t<q>;
    using QUANTIZED_VECTOR_TYPE = QuantizedVectorType_t<q>;
    using DATA_TYPE = DataType_t<q>;
    using INDEX_TYPE = IndexPDXIVF<q>;
    using VECTORGROUP_TYPE = Vectorgroup<q>;
    using KNNCandidate_t = KNNCandidate<q>;
    using VectorComparator_t = VectorComparator<q>;

public:
    ADSamplingSearcher(INDEX_TYPE &pdx_index,
                         size_t ivf_nprobe, float epsilon0, Eigen::MatrixXf matrix,
                         DimensionsOrder dimension_order)
            : PDXearch<q>(pdx_index,
                           ivf_nprobe,
                           1,
                           dimension_order),
              epsilon0(epsilon0), matrix(std::move(matrix)) {
        // Initialize ratios for the dataset
        ratios.resize(this->pdx_data.num_dimensions);
        for (size_t i = 0; i < this->pdx_data.num_dimensions; ++i) {
            ratios[i] = GetRatio(i);
        }
    }

    void SetEpsilon0(float epsilon0) {
        ADSamplingSearcher::epsilon0 = epsilon0;
    }

    void SetMatrix(const Eigen::MatrixXf &matrix) {
        ADSamplingSearcher::matrix = matrix;
    }

    inline void GetPruningThreshold(uint32_t k, std::priority_queue<KNNCandidate_t , std::vector<KNNCandidate_t>, VectorComparator_t> &heap) override {
        float ratio = this->current_dimension_idx == this->pdx_data.num_dimensions ? 1 : ratios[this->current_dimension_idx];
        this->pruning_threshold = heap.top().distance * ratio * this->current_scaling_factor;
    }

    void PreprocessQuery(float *raw_query, float * query) override {
        //memcpy((void *) query, (void *) raw_query, this->pdx_data.num_dimensions * sizeof(QUANTIZED_VECTOR_TYPE));
        Multiply(raw_query, query, this->pdx_data.num_dimensions);
    }

private:
    float epsilon0 = 2.1;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix;

    inline float GetRatio(const size_t &visited_dimensions) {
        if(visited_dimensions == (int) this->pdx_data.num_dimensions) {
            return 1.0;
        }
        return 1.0 * visited_dimensions / ((int) this->pdx_data.num_dimensions) *
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
