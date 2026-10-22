#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "benchmark_utils.hpp"
#include "pdx/indexes/ivf_tree.hpp"
#include "pdx/indexes/ivf_vanilla.hpp"
#include "pdx/profiler.hpp"
#include "pdx/utils.hpp"

// Builds on the first n - n_append rows and appends the rest in batches of batch_size: 1 calls
// Append per row, more calls BulkAppend, 0 builds on all rows instead (the reference).
template <typename IndexT>
void RunBenchmark(
    const RawDatasetInfo& info,
    const std::string& dataset,
    const std::string& algorithm,
    const float* data,
    const float* queries,
    const size_t batch_size,
    const size_t n_append,
    const uint32_t n_threads,
    const uint32_t saq_bits_per_dimension
) {
    const size_t d = info.num_dimensions;
    const size_t n = info.num_embeddings;
    const size_t n_queries = info.num_queries;
    const uint8_t knn = BenchmarkUtils::KNN;
    const std::string results_path = BENCHMARK_UTILS.RESULTS_DIR_PATH + "BULK_APPEND_PDX.csv";
    const size_t n_build = batch_size == 0 ? n : n - n_append;

    PDX::PDXIndexConfig index_config{
        .num_dimensions = static_cast<uint32_t>(d),
        .distance_metric = info.distance_metric,
        .seed = 42,
        .normalize = true,
        .sampling_fraction = 1.0f,
        .n_threads = n_threads,
    };
    index_config.saq_bits_per_dimension = saq_bits_per_dimension;
    std::vector<size_t> row_ids(n);
    for (size_t i = 0; i < n; i++) {
        row_ids[i] = i;
    }

    TicToc clock;
    IndexT pdx_index(index_config);
    clock.Tic();
    pdx_index.BuildIndex(row_ids.data(), data, n_build);
    clock.Toc();
    std::cout << "Build rows: " << n_build << "\n";
    std::cout << "Build time: " << clock.GetMilliseconds() << " ms\n";

    PDX::Profiler::Get().Reset();
    if (batch_size > 0) {
        clock.Reset();
        clock.Tic();
        if (batch_size == 1) {
            for (size_t row_id = n_build; row_id < n; row_id++) {
                pdx_index.Append(row_id, data + row_id * d);
            }
        } else {
            for (size_t first = n_build; first < n; first += batch_size) {
                const size_t count = std::min(batch_size, n - first);
                pdx_index.BulkAppend(row_ids.data() + first, data + first * d, count);
            }
        }
        clock.Toc();
        std::cout << "Append rows: " << n_append << "\n";
        std::cout << "Append time: " << clock.GetMilliseconds() << " ms\n";
        std::cout << "Appends per second: " << std::fixed << std::setprecision(1)
                  << n_append / (clock.GetMilliseconds() / 1000.0) << "\n";
    }

    uint32_t largest_cluster = 0;
    for (uint32_t c = 0; c < pdx_index.GetNumClusters(); c++) {
        largest_cluster = std::max(largest_cluster, pdx_index.GetClusterSize(c));
    }
    std::cout << "Clusters: " << pdx_index.GetNumClusters() << "\n";
    std::cout << "Mean cluster size: " << std::fixed << std::setprecision(1)
              << static_cast<double>(n) / pdx_index.GetNumClusters() << "\n";
    std::cout << "Largest cluster: " << largest_cluster << "\n";
    std::cout << "Index in-memory size: " << std::fixed << std::setprecision(2)
              << static_cast<double>(pdx_index.GetInMemorySizeInBytes()) / (1024.0 * 1024.0)
              << " MB\n";
    PDX::Profiler::Get().PrintHierarchical();

    const std::string gt_path =
        BenchmarkUtils::GROUND_TRUTH_DATA + info.pdx_dataset_name + "_100_norm";
    auto gt_buffer = MmapFile(gt_path);
    const auto* int_ground_truth = reinterpret_cast<const uint32_t*>(gt_buffer.get());
    for (const size_t nprobe : {16, 32, 64, 128}) {
        pdx_index.SetNProbe(nprobe);
        float recalls = 0;
        for (size_t q = 0; q < n_queries; ++q) {
            auto result = pdx_index.Search(queries + q * d, knn);
            BenchmarkUtils::VerifyResult<true>(recalls, result, knn, int_ground_truth, q);
        }
        std::vector<PhasesRuntime> runtimes(n_queries);
        TicToc search_clock;
        for (size_t q = 0; q < n_queries; ++q) {
            search_clock.Reset();
            search_clock.Tic();
            pdx_index.Search(queries + q * d, knn);
            search_clock.Toc();
            runtimes[q] = {search_clock.accum_time};
        }
        double total_ms = 0;
        for (const auto& runtime : runtimes) {
            total_ms += static_cast<double>(runtime.end_to_end) / 1e6;
        }
        std::cout << "nprobe " << nprobe << ": recall " << std::setprecision(4)
                  << recalls / static_cast<float>(n_queries) << ", avg latency "
                  << total_ms / static_cast<double>(n_queries) << " ms\n";
        BenchmarkMetadata results_metadata = {
            dataset,
            algorithm,
            1,
            n_queries,
            nprobe,
            knn,
            recalls,
        };
        BenchmarkUtils::SaveResults(runtimes, results_path, results_metadata);
    }
}

template <typename T>
struct Tag {
    using type = T;
};

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset> <index_type> <batch_size> [n_append] [n_threads]\n";
        std::cerr << "batch_size: 1 = Append per row, 0 = build on all rows (reference)\n";
        std::cerr
            << "Index types: pdx_f32, pdx_u8, pdx_tree_f32, pdx_tree_u8, pdx_saq3, pdx_saq4\n";
        return 1;
    }
    const std::string dataset = argv[1];
    const std::string index_type = argv[2];
    const size_t batch_size = std::stoull(argv[3]);
    auto it = RAW_DATASET_PARAMS.find(dataset);
    if (it == RAW_DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset: " << dataset << "\n";
        return 1;
    }
    const auto& info = it->second;
    const size_t n = info.num_embeddings;
    const size_t d = info.num_dimensions;
    const size_t n_queries = info.num_queries;
    const size_t n_append = (argc > 4) ? std::stoull(argv[4]) : n / 2;
    const auto n_threads = static_cast<uint32_t>((argc > 5) ? std::stoul(argv[5]) : 0);
    if (n_append == 0 || n_append >= n) {
        std::cerr << "n_append must be in (0, " << n << ")\n";
        return 1;
    }

    std::cout << "==> PDX Bulk Append Benchmark\n";
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "Index type: " << index_type << "\n";
    std::cout << "Batch size: " << batch_size << "\n";
    std::cout << "Threads: " << n_threads << "\n";

    std::vector<float> data(n * d);
    {
        const std::string data_path = RAW_DATA_DIR + "/data_" + dataset + ".bin";
        std::ifstream file(data_path, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open " << data_path << "\n";
            return 1;
        }
        file.read(reinterpret_cast<char*>(data.data()), n * d * sizeof(float));
    }
    std::vector<float> queries(n_queries * d);
    {
        const std::string query_path = RAW_DATA_DIR + "/data_" + dataset + "_test.bin";
        std::ifstream file(query_path, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open " << query_path << "\n";
            return 1;
        }
        file.read(reinterpret_cast<char*>(queries.data()), n_queries * d * sizeof(float));
    }

    const std::string algorithm = "bulk_append_" + index_type + "_b" + std::to_string(batch_size) +
                                  "_t" + std::to_string(n_threads);
    const auto run = [&](auto tag,
                         uint32_t saq_bits_per_dimension =
                             PDX::PDXIndexConfig{}.saq_bits_per_dimension) {
        RunBenchmark<typename decltype(tag)::type>(
            info,
            dataset,
            algorithm,
            data.data(),
            queries.data(),
            batch_size,
            n_append,
            n_threads,
            saq_bits_per_dimension
        );
    };
    if (index_type == "pdx_f32") {
        run(Tag<PDX::PDXIndexF32>{});
    } else if (index_type == "pdx_u8") {
        run(Tag<PDX::PDXIndexU8>{});
    } else if (index_type == "pdx_tree_f32") {
        run(Tag<PDX::PDXTreeIndexF32>{});
    } else if (index_type == "pdx_tree_u8") {
        run(Tag<PDX::PDXTreeIndexU8>{});
    } else if (index_type == "pdx_saq3" || index_type == "pdx_saq4") {
        run(Tag<PDX::PDXIndexSAQ>{}, index_type == "pdx_saq3" ? 3 : 4);
    } else {
        std::cerr << "Unknown index type: " << index_type << "\n";
        return 1;
    }
    return 0;
}
