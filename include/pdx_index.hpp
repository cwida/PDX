#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "common.hpp"
#include "ivf_wrapper.hpp"
#include "pruners/adsampling.hpp"
#include "pdxearch.hpp"
#include "utils/file_reader.hpp"

namespace PDX {

template <PDX::Quantization Q>
class PDXIndex {
public:
	using embedding_storage_t = PDX::pdx_data_t<Q>;

private:
	std::unique_ptr<char[]> matrix_buffer;

	PDX::IndexPDXIVF<Q> index;
	std::unique_ptr<PDX::ADSamplingPruner> pruner;
	std::unique_ptr<PDX::PDXearch<Q>> searcher;

public:
	PDXIndex() = default;

	void Restore(const std::string &index_path, const std::string &matrix_path) {
		index.Restore(index_path);

		matrix_buffer = MmapFile(matrix_path);
		auto *matrix = reinterpret_cast<float *>(matrix_buffer.get());

		pruner = std::make_unique<PDX::ADSamplingPruner>(index.num_dimensions, matrix);
		searcher = std::make_unique<PDX::PDXearch<Q>>(index, *pruner);
	}

	std::vector<PDX::KNNCandidate> Search(const float *query_embedding, size_t knn) const {
		return searcher->Search(query_embedding, knn);
	}

	void SetNProbe(uint32_t n_probe) const {
		searcher->SetNProbe(n_probe);
	}

	const PDX::PDXearch<Q> &GetSearcher() const {
		return *searcher;
	}

	uint32_t GetNumDimensions() const {
		return index.num_dimensions;
	}

	uint32_t GetNumClusters() const {
		return index.num_clusters;
	}

	// === CreateIndex path (DuckDB integration) — commented out for now ===
	// Requires: D_ASSERT, make_uniq, row_t, Vector, FlatVector, NotImplementedException,
	//           clustering.hpp, db_mock/predicate_evaluator.hpp
	//
	// void CreateIndex(const row_t *const row_ids, const float *const embeddings, const size_t num_embeddings);
	// std::unique_ptr<std::vector<row_t>> Search(const float *query_embedding, size_t limit, uint32_t n_probe) const;
	// PDX::PredicateEvaluator CreatePredicateEvaluator(...) const;
	// std::unique_ptr<std::vector<row_t>> FilteredSearch(...) const;
};

template <PDX::Quantization Q>
class PDXTreeIndex {
public:
	using embedding_storage_t = PDX::pdx_data_t<Q>;

private:
	std::unique_ptr<char[]> matrix_buffer;

	PDX::IndexPDXIVF2<Q> index;
	std::unique_ptr<PDX::ADSamplingPruner> pruner;
	std::unique_ptr<PDX::PDXearch<Q>> searcher;
	std::unique_ptr<PDX::PDXearch<F32>> top_level_searcher;

public:
	PDXTreeIndex() = default;

	void Restore(const std::string &index_path, const std::string &matrix_path) {
		index.Restore(index_path);

		matrix_buffer = MmapFile(matrix_path);
		auto *matrix = reinterpret_cast<float *>(matrix_buffer.get());

		pruner = std::make_unique<PDX::ADSamplingPruner>(index.num_dimensions, matrix);
		searcher = std::make_unique<PDX::PDXearch<Q>>(index, *pruner);
		top_level_searcher = std::make_unique<PDX::PDXearch<F32>>(index.l0, *pruner);
	}

	std::vector<PDX::KNNCandidate> Search(const float *query_embedding, size_t knn) const {
		auto n_probe_top_level = GetTopLevelNumClusters();
		// if (searcher->GetNProbe() < GetNumClusters() / 2){
		// 	// We confidently prune half of the meso-clusters only if the user wants to
        //     // visit less than half of the available clusters
		// 	n_probe_top_level /= 2;
		// }
		top_level_searcher->SetNProbe(n_probe_top_level);
		auto top_level_results = top_level_searcher->Search(query_embedding, searcher->GetNProbe());

		std::vector<uint32_t> top_level_indexes(top_level_results.size());
		for (size_t i = 0; i < top_level_results.size(); i++) {
			top_level_indexes[i] = top_level_results[i].index;
		}
		searcher->SetClusterAccessOrder(top_level_indexes);

		return searcher->Search(query_embedding, knn);
	}

	void SetNProbe(uint32_t n_probe) const {
		searcher->SetNProbe(n_probe);
	}

	const PDX::PDXearch<Q> &GetSearcher() const {
		return *searcher;
	}

	uint32_t GetNumDimensions() const {
		return index.num_dimensions;
	}

	uint32_t GetNumClusters() const {
		return index.num_clusters;
	}

	uint32_t GetTopLevelNumClusters() const {
		return index.l0.num_clusters;
	}

	// === CreateIndex path (DuckDB integration) — commented out for now ===
	// Requires: D_ASSERT, make_uniq, row_t, Vector, FlatVector, NotImplementedException,
	//           clustering.hpp, db_mock/predicate_evaluator.hpp
	//
	// void CreateIndex(const row_t *const row_ids, const float *const embeddings, const size_t num_embeddings);
	// std::unique_ptr<std::vector<row_t>> Search(const float *query_embedding, size_t limit, uint32_t n_probe) const;
	// PDX::PredicateEvaluator CreatePredicateEvaluator(...) const;
	// std::unique_ptr<std::vector<row_t>> FilteredSearch(...) const;
};

using PDXIndexF32 = PDXIndex<PDX::F32>;
using PDXIndexU8 = PDXIndex<PDX::U8>;
using PDXTreeIndexF32 = PDXTreeIndex<PDX::F32>;
using PDXTreeIndexU8 = PDXTreeIndex<PDX::U8>;

} // namespace PDX
