#pragma once

#include <cstdint>
#include <cstdio>
#include <memory>
#include <cassert>
#include <queue>
#include <random>
#include "common.hpp"
#include "ivf_wrapper.hpp"
#include "quantizers/scalar.hpp"
#include "pruners/adsampling.hpp"


namespace PDX {

// Generate a rotation matrix suitable for PDXearch's ADSampling pruning algorithm.
[[nodiscard]] inline std::unique_ptr<float[]> GenerateRandomRotationMatrix(const size_t num_dimensions, const int32_t seed) {
	auto rotation_matrix = std::make_unique<float[]>(num_dimensions * num_dimensions);

	std::mt19937 gen(seed);
	std::normal_distribution<float> normal_dist;

	Eigen::MatrixXf random_matrix {
	    Eigen::MatrixXf::Zero(static_cast<Eigen::Index>(num_dimensions), static_cast<Eigen::Index>(num_dimensions))};
	for (size_t i = 0; i < num_dimensions; ++i) {
		for (size_t j = 0; j < num_dimensions; ++j) {
			random_matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = normal_dist(gen);
		}
	}

	const Eigen::HouseholderQR<Eigen::MatrixXf> qr {random_matrix};
	const Eigen::MatrixXf transformation_matrix {qr.householderQ()};

	for (size_t i = 0; i < num_dimensions; ++i) {
		for (size_t j = 0; j < num_dimensions; ++j) {
			rotation_matrix[i * num_dimensions + j] =
			    transformation_matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
		}
	}

	return rotation_matrix;
}

// Store the embeddings into this cluster's preallocated buffers in the transposed PDX layout.
//
// See the README of the following for a description of the PDX layout:
// https://github.com/cwida/pdx
template <PDX::Quantization q, typename T>
inline void StoreClusterEmbeddings(typename PDX::IndexPDXIVF<q>::cluster_t &cluster, const PDX::IndexPDXIVF<q> &index,
                                   const T *embeddings, const size_t num_embeddings);

template <>
inline void
StoreClusterEmbeddings<PDX::Quantization::F32, float>(PDX::IndexPDXIVF<PDX::Quantization::F32>::cluster_t &cluster,
                                                      const PDX::IndexPDXIVF<PDX::Quantization::F32> &index,
                                                      const float *const embeddings, const size_t num_embeddings) {
	// Store the cluster's data using the transposed PDX layout for float32 as described in:
	// https://github.com/cwida/pdx?tab=readme-ov-file#the-data-layout
	using matrix_t = PDX::eigen_matrix_t;
	using h_matrix_t = Eigen::Matrix<float, Eigen::Dynamic, PDX::H_DIM_SIZE, Eigen::RowMajor>;

	const auto vertical_d = index.num_vertical_dimensions;
	const auto horizontal_d = index.num_horizontal_dimensions;

	Eigen::Map<const matrix_t> in(embeddings, num_embeddings, index.num_dimensions);

	// Vertical block: transpose the first vertical_d columns into dimension-major order.
	Eigen::Map<matrix_t> out(cluster.data, vertical_d, num_embeddings);
	out.noalias() = in.leftCols(vertical_d).transpose();

	// Horizontal blocks: copy H_DIM_SIZE columns at a time, keeping each embedding's values contiguous.
	float *horizontal_out = cluster.data + num_embeddings * vertical_d;
	for (size_t j = 0; j < horizontal_d; j += PDX::H_DIM_SIZE) {
		Eigen::Map<h_matrix_t> out_h(horizontal_out, num_embeddings, PDX::H_DIM_SIZE);
		out_h.noalias() = in.block(0, vertical_d + j, num_embeddings, PDX::H_DIM_SIZE);
		horizontal_out += num_embeddings * PDX::H_DIM_SIZE;
	}
}

template <>
inline void
StoreClusterEmbeddings<PDX::Quantization::U8, uint8_t>(PDX::IndexPDXIVF<PDX::Quantization::U8>::cluster_t &cluster,
                                                       const PDX::IndexPDXIVF<PDX::Quantization::U8> &index,
                                                       const uint8_t *const embeddings, const size_t num_embeddings) {
	// Store the cluster's data using the transposed PDX layout for U8.
	// The vertical block uses interleaving: for each group of U8_INTERLEAVE_SIZE consecutive dimensions,
	// each vector's values are stored contiguously. This enables NEON vdotq_u32 / AVX2 processing.
	using u8_matrix_t = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using u8_v_matrix_t = Eigen::Matrix<uint8_t, Eigen::Dynamic, PDX::U8_INTERLEAVE_SIZE, Eigen::RowMajor>;
	using u8_h_matrix_t = Eigen::Matrix<uint8_t, Eigen::Dynamic, PDX::H_DIM_SIZE, Eigen::RowMajor>;

	const auto vertical_d = index.num_vertical_dimensions;
	const auto horizontal_d = index.num_horizontal_dimensions;

	Eigen::Map<const u8_matrix_t> in(embeddings, num_embeddings, index.num_dimensions);

	// Vertical dimensions (interleaved): copy U8_INTERLEAVE_SIZE columns at a time into row-major blocks.
	size_t dim = 0;
	for (; dim + PDX::U8_INTERLEAVE_SIZE <= vertical_d; dim += PDX::U8_INTERLEAVE_SIZE) {
		Eigen::Map<u8_v_matrix_t> out_v(cluster.data + dim * num_embeddings, num_embeddings, PDX::U8_INTERLEAVE_SIZE);
		out_v.noalias() = in.block(0, dim, num_embeddings, PDX::U8_INTERLEAVE_SIZE);
	}
	// Compact tail for remaining vertical dimensions (< U8_INTERLEAVE_SIZE).
	if (dim < vertical_d) {
		auto remaining = static_cast<Eigen::Index>(vertical_d - dim);
		Eigen::Map<u8_matrix_t> out_v(cluster.data + dim * num_embeddings, num_embeddings, remaining);
		out_v.noalias() = in.block(0, dim, num_embeddings, remaining);
	}

	// Horizontal blocks: copy H_DIM_SIZE columns at a time, keeping each embedding's values contiguous.
	uint8_t *horizontal_out = cluster.data + num_embeddings * vertical_d;
	for (size_t j = 0; j < horizontal_d; j += PDX::H_DIM_SIZE) {
		Eigen::Map<u8_h_matrix_t> out_h(horizontal_out, num_embeddings, PDX::H_DIM_SIZE);
		out_h.noalias() = in.block(0, vertical_d + j, num_embeddings, PDX::H_DIM_SIZE);
		horizontal_out += num_embeddings * PDX::H_DIM_SIZE;
	}
}

class EmbeddingPreprocessor {
private:
	// For rotation matrix multiplication.
	PDX::ADSamplingPruner pruner;
	// For normalization.
	PDX::Quantizer quantizer;
	const size_t num_dimensions;

public:
	explicit EmbeddingPreprocessor(const size_t num_dimensions, const float *const rotation_matrix)
	    : pruner(num_dimensions, rotation_matrix), quantizer(num_dimensions), num_dimensions(num_dimensions) {
	}

	// Warning: modifies the input_embedding.
	void PreprocessEmbedding(float *const input_embedding, float *const output_embedding, const bool normalize) const {
		// In-place normalization.
		if (normalize) {
			quantizer.NormalizeQuery(input_embedding, input_embedding);
		}
		pruner.PreprocessQuery(input_embedding, output_embedding);
	}

	// Warning: modifies the input_embeddings.
	void PreprocessEmbeddings(float *const input_embeddings, float *const output_embeddings,
	                          const size_t num_embeddings, const bool normalize) const {
		// In-place normalization.
		if (normalize) {
			for (size_t i = 0; i < num_embeddings; i++) {
				quantizer.NormalizeQuery(input_embeddings + i * num_dimensions, input_embeddings + i * num_dimensions);
			}
		}
		pruner.PreprocessEmbeddings(input_embeddings, output_embeddings, num_embeddings);
	}
};

[[nodiscard]] inline constexpr bool DistanceMetricRequiresNormalization(const PDX::DistanceMetric distance_metric) {
	return distance_metric == PDX::DistanceMetric::COSINE || distance_metric == PDX::DistanceMetric::IP;
}

} // namespace PDX