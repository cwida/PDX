#pragma once

#include "pdx/common.hpp"
#include "pdx/ivf_wrapper.hpp"

namespace PDX {

// Store the embeddings into this cluster's preallocated buffers in the transposed PDX layout.
//
// See the README of the following for a description of the PDX layout:
// https://github.com/cwida/pdx
template <PDX::Quantization q, typename T>
inline void StoreClusterEmbeddings(typename PDX::IVF<q>::cluster_t &cluster, const PDX::IVF<q> &index,
                                   const T *embeddings, const size_t num_embeddings);

template <>
inline void
StoreClusterEmbeddings<PDX::Quantization::F32, float>(PDX::IVF<PDX::Quantization::F32>::cluster_t &cluster,
                                                      const PDX::IVF<PDX::Quantization::F32> &index,
                                                      const float *const embeddings, const size_t num_embeddings) {
	using matrix_t = PDX::eigen_matrix_t;
	using h_matrix_t = Eigen::Matrix<float, Eigen::Dynamic, PDX::H_DIM_SIZE, Eigen::RowMajor>;

	const auto vertical_d = index.num_vertical_dimensions;
	const auto horizontal_d = index.num_horizontal_dimensions;

	Eigen::Map<const matrix_t> in(embeddings, num_embeddings, index.num_dimensions);

	Eigen::Map<matrix_t> out(cluster.data, vertical_d, num_embeddings);
	out.noalias() = in.leftCols(vertical_d).transpose();

	float *horizontal_out = cluster.data + num_embeddings * vertical_d;
	for (size_t j = 0; j < horizontal_d; j += PDX::H_DIM_SIZE) {
		Eigen::Map<h_matrix_t> out_h(horizontal_out, num_embeddings, PDX::H_DIM_SIZE);
		out_h.noalias() = in.block(0, vertical_d + j, num_embeddings, PDX::H_DIM_SIZE);
		horizontal_out += num_embeddings * PDX::H_DIM_SIZE;
	}
}

template <>
inline void
StoreClusterEmbeddings<PDX::Quantization::U8, uint8_t>(PDX::IVF<PDX::Quantization::U8>::cluster_t &cluster,
                                                       const PDX::IVF<PDX::Quantization::U8> &index,
                                                       const uint8_t *const embeddings, const size_t num_embeddings) {
	using u8_matrix_t = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using u8_v_matrix_t = Eigen::Matrix<uint8_t, Eigen::Dynamic, PDX::U8_INTERLEAVE_SIZE, Eigen::RowMajor>;
	using u8_h_matrix_t = Eigen::Matrix<uint8_t, Eigen::Dynamic, PDX::H_DIM_SIZE, Eigen::RowMajor>;

	const auto vertical_d = index.num_vertical_dimensions;
	const auto horizontal_d = index.num_horizontal_dimensions;

	Eigen::Map<const u8_matrix_t> in(embeddings, num_embeddings, index.num_dimensions);

	size_t dim = 0;
	for (; dim + PDX::U8_INTERLEAVE_SIZE <= vertical_d; dim += PDX::U8_INTERLEAVE_SIZE) {
		Eigen::Map<u8_v_matrix_t> out_v(cluster.data + dim * num_embeddings, num_embeddings, PDX::U8_INTERLEAVE_SIZE);
		out_v.noalias() = in.block(0, dim, num_embeddings, PDX::U8_INTERLEAVE_SIZE);
	}
	if (dim < vertical_d) {
		auto remaining = static_cast<Eigen::Index>(vertical_d - dim);
		Eigen::Map<u8_matrix_t> out_v(cluster.data + dim * num_embeddings, num_embeddings, remaining);
		out_v.noalias() = in.block(0, dim, num_embeddings, remaining);
	}

	uint8_t *horizontal_out = cluster.data + num_embeddings * vertical_d;
	for (size_t j = 0; j < horizontal_d; j += PDX::H_DIM_SIZE) {
		Eigen::Map<u8_h_matrix_t> out_h(horizontal_out, num_embeddings, PDX::H_DIM_SIZE);
		out_h.noalias() = in.block(0, vertical_d + j, num_embeddings, PDX::H_DIM_SIZE);
		horizontal_out += num_embeddings * PDX::H_DIM_SIZE;
	}
}

} // namespace PDX
