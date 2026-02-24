#pragma once

#include <cstdint>
#include <cstring>
#include <cassert>
#include <ostream>
#include <vector>
#include <memory>
#include "pdx/common.hpp"
#include "pdx/utils.hpp"

namespace PDX {

template <Quantization Q>
class IndexPDXIVF {
public:
	using cluster_t = Cluster<Q>;
	using data_t = pdx_data_t<Q>;

	uint32_t num_dimensions {};
	uint64_t total_num_embeddings {};
	uint32_t num_clusters {};
	uint32_t num_vertical_dimensions {};
	uint32_t num_horizontal_dimensions {};
	std::vector<cluster_t> clusters;
	bool is_normalized {};
	std::vector<float> centroids;

	// U8-specific quantization parameters
	float quantization_scale = 1.0f;
	float quantization_scale_squared = 1.0f;
	float inverse_quantization_scale_squared = 1.0f;
	float quantization_base = 0.0f;

	IndexPDXIVF() = default;
	~IndexPDXIVF() = default;
	IndexPDXIVF(IndexPDXIVF &&) = default;
	IndexPDXIVF &operator=(IndexPDXIVF &&) = default;

	IndexPDXIVF(uint32_t num_dimensions, uint64_t total_num_embeddings, uint32_t num_clusters, bool is_normalized)
	    : num_dimensions(num_dimensions), total_num_embeddings(total_num_embeddings), num_clusters(num_clusters),
	      num_vertical_dimensions(GetPDXDimensionSplit(num_dimensions).vertical_dimensions),
	      num_horizontal_dimensions(GetPDXDimensionSplit(num_dimensions).horizontal_dimensions),
	      is_normalized(is_normalized) {
		clusters.reserve(num_clusters);
	}

	IndexPDXIVF(uint32_t num_dimensions, uint64_t total_num_embeddings, uint32_t num_clusters, bool is_normalized,
	            float quantization_scale, float quantization_base)
	    : num_dimensions(num_dimensions), total_num_embeddings(total_num_embeddings), num_clusters(num_clusters),
	      num_vertical_dimensions(GetPDXDimensionSplit(num_dimensions).vertical_dimensions),
	      num_horizontal_dimensions(GetPDXDimensionSplit(num_dimensions).horizontal_dimensions),
	      is_normalized(is_normalized), quantization_scale(quantization_scale),
	      quantization_scale_squared(quantization_scale * quantization_scale),
	      inverse_quantization_scale_squared(1.0f / (quantization_scale * quantization_scale)),
	      quantization_base(quantization_base) {
		clusters.reserve(num_clusters);
	}

	void Load(char *input) {
		char *next_value = input;
		num_dimensions = ((uint32_t *) input)[0];
		num_vertical_dimensions = ((uint32_t *) input)[1];
		num_horizontal_dimensions = ((uint32_t *) input)[2];

		next_value += sizeof(uint32_t) * 3;
		num_clusters = ((uint32_t *) next_value)[0];
		next_value += sizeof(uint32_t);
		auto *nums_embeddings = (uint32_t *) next_value;
		next_value += num_clusters * sizeof(uint32_t);
		clusters.reserve(num_clusters);
		for (size_t i = 0; i < num_clusters; ++i) {
			clusters.emplace_back(nums_embeddings[i], num_dimensions);
			memcpy(clusters[i].data, next_value, sizeof(data_t) * clusters[i].num_embeddings * num_dimensions);
			next_value += sizeof(data_t) * clusters[i].num_embeddings * num_dimensions;
		}
		for (size_t i = 0; i < num_clusters; ++i) {
			memcpy(clusters[i].indices, next_value, sizeof(uint32_t) * clusters[i].num_embeddings);
			next_value += sizeof(uint32_t) * clusters[i].num_embeddings;
		}

		is_normalized = ((char *) next_value)[0];
		next_value += sizeof(char);

		centroids.resize(num_clusters * num_dimensions);
		memcpy(centroids.data(), (float *) next_value, sizeof(float) * num_clusters * num_dimensions);
		next_value += sizeof(float) * num_clusters * num_dimensions;

		if constexpr (Q == U8) {
			quantization_base = ((float *) next_value)[0];
			next_value += sizeof(float);
			quantization_scale = ((float *) next_value)[0];
			next_value += sizeof(float);
			quantization_scale_squared = quantization_scale * quantization_scale;
			inverse_quantization_scale_squared = 1.0f / quantization_scale_squared;
		}
	}

	void Save(std::ostream &out) const {
		out.write(reinterpret_cast<const char *>(&num_dimensions), sizeof(uint32_t));
		out.write(reinterpret_cast<const char *>(&num_vertical_dimensions), sizeof(uint32_t));
		out.write(reinterpret_cast<const char *>(&num_horizontal_dimensions), sizeof(uint32_t));
		out.write(reinterpret_cast<const char *>(&num_clusters), sizeof(uint32_t));

		for (size_t i = 0; i < num_clusters; ++i) {
			out.write(reinterpret_cast<const char *>(&clusters[i].num_embeddings), sizeof(uint32_t));
		}
		for (size_t i = 0; i < num_clusters; ++i) {
			out.write(reinterpret_cast<const char *>(clusters[i].data),
			          sizeof(data_t) * clusters[i].num_embeddings * num_dimensions);
		}
		for (size_t i = 0; i < num_clusters; ++i) {
			out.write(reinterpret_cast<const char *>(clusters[i].indices),
			          sizeof(uint32_t) * clusters[i].num_embeddings);
		}

		char norm = is_normalized;
		out.write(&norm, sizeof(char));

		out.write(reinterpret_cast<const char *>(centroids.data()),
		          sizeof(float) * num_clusters * num_dimensions);

		if constexpr (Q == U8) {
			out.write(reinterpret_cast<const char *>(&quantization_base), sizeof(float));
			out.write(reinterpret_cast<const char *>(&quantization_scale), sizeof(float));
		}
	}
};

template <Quantization Q>
class IndexPDXIVF2 : public IndexPDXIVF<Q> {
public:
	using data_t = pdx_data_t<Q>;

	IndexPDXIVF<F32> l0; // Meso clusters

	IndexPDXIVF2() = default;
	~IndexPDXIVF2() = default;
	IndexPDXIVF2(IndexPDXIVF2 &&) = default;
	IndexPDXIVF2 &operator=(IndexPDXIVF2 &&) = default;

	IndexPDXIVF2(uint32_t num_dimensions, uint64_t total_num_embeddings, uint32_t num_clusters, bool is_normalized)
	    : IndexPDXIVF<Q>(num_dimensions, total_num_embeddings, num_clusters, is_normalized) {}

	IndexPDXIVF2(uint32_t num_dimensions, uint64_t total_num_embeddings, uint32_t num_clusters, bool is_normalized,
	             float quantization_scale, float quantization_base)
	    : IndexPDXIVF<Q>(num_dimensions, total_num_embeddings, num_clusters, is_normalized,
	                     quantization_scale, quantization_base) {}

	void Load(char *input) {
		char *next_value = input;

		// Header
		uint32_t dims = ((uint32_t *) input)[0];
		uint32_t v_dims = ((uint32_t *) input)[1];
		uint32_t h_dims = ((uint32_t *) input)[2];
		next_value += sizeof(uint32_t) * 3;

		uint32_t n_clusters_l1 = ((uint32_t *) next_value)[0];
		next_value += sizeof(uint32_t);
		uint32_t n_clusters_l0 = ((uint32_t *) next_value)[0];
		next_value += sizeof(uint32_t);

		// === L0 (meso-clusters, always F32) ===
		l0.num_dimensions = dims;
		l0.num_vertical_dimensions = v_dims;
		l0.num_horizontal_dimensions = h_dims;
		l0.num_clusters = n_clusters_l0;

		auto *nums_embeddings_l0 = (uint32_t *) next_value;
		next_value += n_clusters_l0 * sizeof(uint32_t);

		l0.clusters.reserve(n_clusters_l0);
		for (size_t i = 0; i < n_clusters_l0; ++i) {
			l0.clusters.emplace_back(nums_embeddings_l0[i], dims);
			memcpy(l0.clusters[i].data, next_value,
			       sizeof(float) * l0.clusters[i].num_embeddings * dims);
			next_value += sizeof(float) * l0.clusters[i].num_embeddings * dims;
		}
		for (size_t i = 0; i < n_clusters_l0; ++i) {
			memcpy(l0.clusters[i].indices, next_value,
			       sizeof(uint32_t) * l0.clusters[i].num_embeddings);
			next_value += sizeof(uint32_t) * l0.clusters[i].num_embeddings;
		}

		// === L1 (data clusters, inherited fields) ===
		this->num_dimensions = dims;
		this->num_vertical_dimensions = v_dims;
		this->num_horizontal_dimensions = h_dims;
		this->num_clusters = n_clusters_l1;

		auto *nums_embeddings_l1 = (uint32_t *) next_value;
		next_value += n_clusters_l1 * sizeof(uint32_t);

		this->clusters.reserve(n_clusters_l1);
		for (size_t i = 0; i < n_clusters_l1; ++i) {
			this->clusters.emplace_back(nums_embeddings_l1[i], dims);
			memcpy(this->clusters[i].data, next_value,
			       sizeof(data_t) * this->clusters[i].num_embeddings * dims);
			next_value += sizeof(data_t) * this->clusters[i].num_embeddings * dims;
		}
		for (size_t i = 0; i < n_clusters_l1; ++i) {
			memcpy(this->clusters[i].indices, next_value,
			       sizeof(uint32_t) * this->clusters[i].num_embeddings);
			next_value += sizeof(uint32_t) * this->clusters[i].num_embeddings;
		}

		// === Shared metadata ===
		bool normalized = ((char *) next_value)[0];
		this->is_normalized = normalized;
		l0.is_normalized = normalized;
		next_value += sizeof(char);

		// === L0 centroids (centroids_pdx from file) ===
		l0.centroids.resize(n_clusters_l0 * dims);
		memcpy(l0.centroids.data(), (float *) next_value,
		       sizeof(float) * n_clusters_l0 * dims);
		next_value += sizeof(float) * n_clusters_l0 * dims;

		// === U8 quantization params ===
		if constexpr (Q == U8) {
			this->quantization_base = ((float *) next_value)[0];
			next_value += sizeof(float);
			this->quantization_scale = ((float *) next_value)[0];
			next_value += sizeof(float);
			this->quantization_scale_squared =
			    this->quantization_scale * this->quantization_scale;
			this->inverse_quantization_scale_squared =
			    1.0f / this->quantization_scale_squared;
		}
	}

	void Save(std::ostream &out) const {
		// Header: dimensions (shared between L0 and L1)
		out.write(reinterpret_cast<const char *>(&this->num_dimensions), sizeof(uint32_t));
		out.write(reinterpret_cast<const char *>(&this->num_vertical_dimensions), sizeof(uint32_t));
		out.write(reinterpret_cast<const char *>(&this->num_horizontal_dimensions), sizeof(uint32_t));

		// Number of clusters: L1 then L0
		out.write(reinterpret_cast<const char *>(&this->num_clusters), sizeof(uint32_t));
		uint32_t n_clusters_l0 = l0.num_clusters;
		out.write(reinterpret_cast<const char *>(&n_clusters_l0), sizeof(uint32_t));

		// === L0 (meso-clusters, always F32) ===
		for (size_t i = 0; i < n_clusters_l0; ++i) {
			out.write(reinterpret_cast<const char *>(&l0.clusters[i].num_embeddings), sizeof(uint32_t));
		}
		for (size_t i = 0; i < n_clusters_l0; ++i) {
			out.write(reinterpret_cast<const char *>(l0.clusters[i].data),
			          sizeof(float) * l0.clusters[i].num_embeddings * this->num_dimensions);
		}
		for (size_t i = 0; i < n_clusters_l0; ++i) {
			out.write(reinterpret_cast<const char *>(l0.clusters[i].indices),
			          sizeof(uint32_t) * l0.clusters[i].num_embeddings);
		}

		// === L1 (data clusters) ===
		for (size_t i = 0; i < this->num_clusters; ++i) {
			out.write(reinterpret_cast<const char *>(&this->clusters[i].num_embeddings), sizeof(uint32_t));
		}
		for (size_t i = 0; i < this->num_clusters; ++i) {
			out.write(reinterpret_cast<const char *>(this->clusters[i].data),
			          sizeof(data_t) * this->clusters[i].num_embeddings * this->num_dimensions);
		}
		for (size_t i = 0; i < this->num_clusters; ++i) {
			out.write(reinterpret_cast<const char *>(this->clusters[i].indices),
			          sizeof(uint32_t) * this->clusters[i].num_embeddings);
		}

		// === Shared metadata ===
		char norm = this->is_normalized;
		out.write(&norm, sizeof(char));

		// L0 centroids
		out.write(reinterpret_cast<const char *>(l0.centroids.data()),
		          sizeof(float) * n_clusters_l0 * this->num_dimensions);

		// === U8 quantization params ===
		if constexpr (Q == U8) {
			out.write(reinterpret_cast<const char *>(&this->quantization_base), sizeof(float));
			out.write(reinterpret_cast<const char *>(&this->quantization_scale), sizeof(float));
		}
	}
};

} // namespace PDX
