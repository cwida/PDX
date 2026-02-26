#include "benchmark_utils.hpp"
#include "pdx/index.hpp"
#include "pdx/utils.hpp"
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>

struct SpecialFilter {
    std::string name;
    uint32_t n_clusters;
    bool plus_one_per_remaining; // PART+ mode: add 1 random row from each remaining cluster
};

static const SpecialFilter SPECIAL_FILTERS[] = {
    {"PART_1", 1, false},
    {"PART_30", 30, false},
    {"PART+_1", 1, true},
};

std::vector<size_t> BuildSpecialFilterRowIds(
    const PDX::IPDXIndex& index,
    const SpecialFilter& filter,
    uint32_t seed
) {
    std::vector<size_t> passing_row_ids;
    uint32_t total_clusters = index.GetNumClusters();
    uint32_t n_full = std::min(filter.n_clusters, total_clusters);

    // All rows from the first n_full clusters
    for (uint32_t c = 0; c < n_full; c++) {
        auto ids = index.GetClusterRowIds(c);
        for (auto id : ids) {
            passing_row_ids.push_back(id);
        }
    }

    // PART+ mode: add 1 random row from each remaining cluster
    if (filter.plus_one_per_remaining) {
        std::mt19937 gen(seed);
        for (uint32_t c = n_full; c < total_clusters; c++) {
            uint32_t cluster_size = index.GetClusterSize(c);
            if (cluster_size == 0) {
                continue;
            }
            auto ids = index.GetClusterRowIds(c);
            std::uniform_int_distribution<uint32_t> dist(0, cluster_size - 1);
            passing_row_ids.push_back(ids[dist(gen)]);
        }
    }

    return passing_row_ids;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset> [index_type] [nprobe]\n";
        std::cerr << "Index types: pdx_f32 (default), pdx_u8, pdx_tree_f32, pdx_tree_u8\n";
        std::cerr << "Available datasets:";
        for (const auto& [name, _] : RAW_DATASET_PARAMS) {
            std::cerr << " " << name;
        }
        std::cerr << "\n";
        return 1;
    }

    std::string arg_dataset = argv[1];
    std::string index_type = "pdx_f32";
    size_t arg_ivf_nprobe = 0;
    if (argc > 2) {
        index_type = argv[2];
    }
    if (argc > 3) {
        arg_ivf_nprobe = atoi(argv[3]);
    }

    std::cout << "==> PDX Special Filtered Benchmark (" << index_type << ")\n";

    std::string ALGORITHM = "special_filtered";
    uint8_t KNN = BenchmarkUtils::KNN;
    size_t NUM_QUERIES;
    size_t NUM_MEASURE_RUNS = BenchmarkUtils::NUM_MEASURE_RUNS;

    std::string index_type_upper = index_type;
    for (auto& c : index_type_upper)
        c = toupper(c);
    std::string RESULTS_PATH =
        BENCHMARK_UTILS.RESULTS_DIR_PATH + index_type_upper + "_SPECIAL_FILTERED.csv";

    for (const auto& [dataset, info] : RAW_DATASET_PARAMS) {
        if (!arg_dataset.empty() && arg_dataset != dataset) {
            continue;
        }

        std::string index_path = BenchmarkUtils::PDX_DATA + dataset + "-" + index_type;
        std::cout << "Loading " << index_path << "...\n";
        auto pdx_index = PDX::LoadPDXIndex(index_path);
        std::cout << "Index in-memory size: " << std::fixed << std::setprecision(2)
                  << static_cast<double>(pdx_index->GetInMemorySizeInBytes()) / (1024.0 * 1024.0)
                  << " MB\n";
        std::cout << "Num clusters: " << pdx_index->GetNumClusters() << "\n";

        // Load queries
        std::unique_ptr<char[]> query_ptr =
            MmapFile(BenchmarkUtils::QUERIES_DATA + info.pdx_dataset_name);
        auto* query = reinterpret_cast<float*>(query_ptr.get());
        NUM_QUERIES = info.num_queries;
        query += 1; // skip number of embeddings header

        for (const auto& filter : SPECIAL_FILTERS) {
            if (filter.n_clusters > pdx_index->GetNumClusters()) {
                std::cout << "Skipping " << filter.name << " (needs " << filter.n_clusters
                          << " clusters, index has " << pdx_index->GetNumClusters() << ")\n";
                continue;
            }

            auto passing_row_ids = BuildSpecialFilterRowIds(*pdx_index, filter, 42);
            std::cout << "\n--- " << filter.name << ": " << passing_row_ids.size()
                      << " passing row IDs ---\n";

            std::vector<size_t> nprobes_to_use;
            if (arg_ivf_nprobe > 0) {
                nprobes_to_use = {arg_ivf_nprobe};
            } else {
                nprobes_to_use.assign(
                    std::begin(BenchmarkUtils::IVF_PROBES), std::end(BenchmarkUtils::IVF_PROBES)
                );
            }

            for (size_t ivf_nprobe : nprobes_to_use) {
                if (pdx_index->GetNumClusters() < ivf_nprobe) {
                    continue;
                }
                if (arg_ivf_nprobe > 0 && ivf_nprobe != arg_ivf_nprobe) {
                    continue;
                }
                std::vector<PhasesRuntime> runtimes;
                runtimes.resize(NUM_MEASURE_RUNS * NUM_QUERIES);
                pdx_index->SetNProbe(ivf_nprobe);

                TicToc clock;
                for (size_t j = 0; j < NUM_MEASURE_RUNS; ++j) {
                    for (size_t l = 0; l < NUM_QUERIES; ++l) {
                        clock.Reset();
                        clock.Tic();
                        pdx_index->FilteredSearch(
                            query + l * pdx_index->GetNumDimensions(), KNN, passing_row_ids
                        );
                        clock.Toc();
                        runtimes[j + l * NUM_MEASURE_RUNS] = {clock.accum_time};
                    }
                }
                BenchmarkMetadata results_metadata = {
                    dataset,
                    ALGORITHM + "_" + filter.name,
                    NUM_MEASURE_RUNS,
                    NUM_QUERIES,
                    ivf_nprobe,
                    KNN,
                    0.0f, // no recall (timing-only)
                    0.0f
                };
                BenchmarkUtils::SaveResults(runtimes, RESULTS_PATH, results_metadata);
            }
        }
    }
    return 0;
}
