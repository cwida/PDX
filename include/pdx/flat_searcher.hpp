#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "pdx/common.hpp"
#include "pdx/distance_computers/base_computers.hpp"
#include "pdx/indexes/flat_core.hpp"
#include "pdx/ivf_searcher.hpp"
#include "pdx/pruners/adsampling.hpp"
#include "pdx/quantizers/scalar.hpp"

namespace PDX {

// Exact search over a Flat block with the horizontal kernel; the Flat counterpart of PDXearch
class FlatSearcher {
  public:
    using distance_computer_t = DistanceComputer<DistanceMetric::L2SQ, F32>;

    Quantizer quantizer;
    ADSamplingPruner& pruner;
    Flat& flat_data;

    FlatSearcher(Flat& data, ADSamplingPruner& pruner)
        : quantizer(data.num_dimensions), pruner(pruner), flat_data(data) {}

    // The block counts as one cluster: the first Next() scans it and the cursor is Done()
    class IterativeSearch final : public IIterativeSearch {
      public:
        IterativeSearch(IterativeSearch&&) noexcept = default;

        size_t Next(size_t n_clusters) override {
            if (done || n_clusters == 0) {
                return 0;
            }
            done = true;
            const Flat& flat_data = searcher->flat_data;
            Heap local_heap;
            if (passing_positions) {
                for (const uint32_t position : *passing_positions) {
                    ScanVector(position, local_heap);
                }
            } else {
                for (uint32_t position = 0; position < flat_data.UsedCapacity(); position++) {
                    if (!flat_data.HasTombstone(position)) {
                        ScanVector(position, local_heap);
                    }
                }
            }
            auto lock = top_k_heap->GetLock();
            Heap& heap = top_k_heap->heap;
            while (!local_heap.empty()) {
                const KNNCandidate candidate = local_heap.top();
                local_heap.pop();
                if (heap.size() < k || candidate.distance < heap.top().distance) {
                    if (heap.size() >= k) {
                        heap.pop();
                    }
                    heap.push(candidate);
                }
            }
            return 1;
        }

        [[nodiscard]] bool Done() const override { return done; }

        [[nodiscard]] size_t ClustersRemaining() const override { return done ? 0 : 1; }

      private:
        friend class FlatSearcher;

        IterativeSearch(
            FlatSearcher& searcher,
            std::unique_ptr<float[]> query,
            uint32_t k,
            TopKHeap& top_k_heap,
            std::unique_ptr<std::vector<uint32_t>> passing_positions
        )
            : searcher(&searcher), top_k_heap(&top_k_heap), k(k), query(std::move(query)),
              passing_positions(std::move(passing_positions)) {}

        void ScanVector(uint32_t position, Heap& heap) const {
            const Flat& flat_data = searcher->flat_data;
            const float distance = distance_computer_t::Horizontal(
                query.get(), flat_data.GetEmbedding(position), flat_data.num_dimensions
            );
            if (heap.size() < k || distance < heap.top().distance) {
                if (heap.size() >= k) {
                    heap.pop();
                }
                heap.push(KNNCandidate{flat_data.indices[position], distance});
            }
        }

        FlatSearcher* searcher;
        TopKHeap* top_k_heap;
        uint32_t k;
        std::unique_ptr<float[]> query;
        std::unique_ptr<std::vector<uint32_t>> passing_positions;
        bool done = false;
    };

    std::vector<KNNCandidate> Search(
        const float* PDX_RESTRICT raw_query,
        size_t k,
        bool is_query_transformed = false
    ) {
        TopKHeap top_k_heap;
        BeginIterativeSearch(raw_query, static_cast<uint32_t>(k), top_k_heap, is_query_transformed)
            .Next(1);
        return BuildResultSetFromHeap(static_cast<uint32_t>(k), top_k_heap.heap);
    }

    std::vector<KNNCandidate> FilteredSearch(
        const float* PDX_RESTRICT raw_query,
        size_t k,
        std::unique_ptr<std::vector<uint32_t>> passing_positions,
        bool is_query_transformed = false
    ) {
        TopKHeap top_k_heap;
        BeginFilteredIterativeSearch(
            raw_query,
            static_cast<uint32_t>(k),
            std::move(passing_positions),
            top_k_heap,
            is_query_transformed
        )
            .Next(1);
        return BuildResultSetFromHeap(static_cast<uint32_t>(k), top_k_heap.heap);
    }

    [[nodiscard]] IterativeSearch BeginIterativeSearch(
        const float* PDX_RESTRICT raw_query,
        uint32_t k,
        TopKHeap& top_k_heap,
        bool is_query_transformed = false
    ) {
        return IterativeSearch(
            *this, PrepareQuery(raw_query, is_query_transformed), k, top_k_heap, nullptr
        );
    }

    [[nodiscard]] IterativeSearch BeginFilteredIterativeSearch(
        const float* PDX_RESTRICT raw_query,
        uint32_t k,
        std::unique_ptr<std::vector<uint32_t>> passing_positions,
        TopKHeap& top_k_heap,
        bool is_query_transformed = false
    ) {
        assert(passing_positions);
        return IterativeSearch(
            *this,
            PrepareQuery(raw_query, is_query_transformed),
            k,
            top_k_heap,
            std::move(passing_positions)
        );
    }

  private:
    std::unique_ptr<float[]> PrepareQuery(
        const float* PDX_RESTRICT raw_query,
        bool is_query_transformed
    ) {
        const size_t d = flat_data.num_dimensions;
        std::unique_ptr<float[]> query(new float[d]);
        if (is_query_transformed) {
            std::copy(raw_query, raw_query + d, query.get());
        } else if (!flat_data.is_normalized) {
            pruner.PreprocessQuery(raw_query, query.get());
        } else {
            std::unique_ptr<float[]> normalized_query(new float[d]);
            quantizer.NormalizeQuery(raw_query, normalized_query.get());
            pruner.PreprocessQuery(normalized_query.get(), query.get());
        }
        return query;
    }
};

} // namespace PDX
