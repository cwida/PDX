// Centroid quantization end to end: a u8 vanilla IVF whose centroids are float32, LVQ8 or LVQ4 (the
// search ranks LVQ centroids with an int8 query), through one workload, then an n_probe x k sweep.
//   BenchmarkCentroidQuantization <dataset> <build|append|churn> <f32|lvq8|lvq4> <results_prefix>
// Workloads, in fractions of the dataset:
//   build:  BUILD 1.0
//   append: BUILD 0.5, APPEND 0.5
//   churn:  BUILD 0.5, APPEND 0.25, DELETE 0.1, APPEND 0.25, DELETE 0.1, APPEND 0.2
// Deleted rows come back under new row ids, so the index ends with every row once and the dataset's
// ground truth applies after remapping the ids.
// The build uses every thread (num_clusters = 4 sqrt(built rows), flat SuperKMeans, 10 iterations).
// Appends and deletes go one row at a time. Queries run one at a time, each timed in two steps:
// BeginIterativeSearch (query rotation + centroid ranking) and Next(n_probe) (the cluster search).
// The query rotation alone is timed apart, on a standalone ADSamplingPruner of the same
// dimensionality. Appends <results_prefix>_phases.csv and <results_prefix>_search.csv.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "benchmark_utils.hpp"
#include "pdx/indexes/ivf_tree.hpp"
#include "pdx/indexes/ivf_vanilla.hpp"
#include "pdx/pruners/adsampling.hpp"
#include "pdx/utils.hpp"

namespace {

using Clock = std::chrono::steady_clock;

const std::vector<size_t> NPROBES = {12, 24, 48, 96};
const std::vector<uint32_t> KS = {10, 20, 100};

std::vector<WorkloadStep> GetWorkload(const std::string& name) {
    if (name == "build") {
        return {{StepType::BUILD, 1.0f}};
    }
    if (name == "append") {
        return {{StepType::BUILD, 0.5f}, {StepType::INSERT, 0.5f}};
    }
    if (name == "churn") {
        return {
            {StepType::BUILD, 0.5f},
            {StepType::INSERT, 0.25f},
            {StepType::DELETE, 0.1f},
            {StepType::INSERT, 0.25f},
            {StepType::DELETE, 0.1f},
            {StepType::INSERT, 0.2f}
        };
    }
    throw std::invalid_argument("Unknown workload: " + name);
}

PDX::CentroidQuantization GetCentroidQuantization(const std::string& name) {
    if (name == "f32") {
        return PDX::CentroidQuantization::F32;
    }
    if (name == "lvq8") {
        return PDX::CentroidQuantization::LVQ8;
    }
    if (name == "lvq4") {
        return PDX::CentroidQuantization::LVQ4;
    }
    throw std::invalid_argument("Unknown centroid quantization: " + name);
}

double ElapsedMs(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

double Percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const auto rank = static_cast<size_t>(std::ceil(fraction * static_cast<double>(values.size())));
    return values[std::min(values.size(), std::max<size_t>(rank, 1)) - 1];
}

double Mean(const std::vector<double>& values) {
    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }
    return sum / static_cast<double>(values.size());
}

std::ofstream OpenCsv(const std::string& path, const std::string& header) {
    const bool exists = std::filesystem::exists(path);
    std::ofstream file(path, std::ios::app);
    if (!exists) {
        file << header << "\n";
    }
    return file;
}

std::vector<float> ReadFloats(const std::string& path, size_t count) {
    std::vector<float> values(count);
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open " + path);
    }
    file.read(
        reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float))
    );
    return values;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset> <build|append|churn> <f32|lvq8|lvq4> <results_prefix>\n";
        return 1;
    }
    const std::string dataset = argv[1];
    const std::string workload_name = argv[2];
    const std::string centroids_name = argv[3];
    const std::string results_prefix = argv[4];
    const auto workload = GetWorkload(workload_name);
    const auto centroid_quantization = GetCentroidQuantization(centroids_name);
    const auto& info = RAW_DATASET_PARAMS.at(dataset);
    const size_t n = info.num_embeddings;
    const size_t d = info.num_dimensions;
    const size_t n_queries = info.num_queries;

    std::cout << "==> " << dataset << " (n=" << n << ", d=" << d << ") workload=" << workload_name
              << " centroids=" << centroids_name << std::endl;
    const auto data = ReadFloats(RAW_DATA_DIR + "/data_" + dataset + ".bin", n * d);
    const auto queries = ReadFloats(RAW_DATA_DIR + "/data_" + dataset + "_test.bin", n_queries * d);

    const size_t n_build =
        static_cast<size_t>(static_cast<double>(n) * workload.front().proportion);
    PDX::PDXIndexConfig config{
        .num_dimensions = static_cast<uint32_t>(d),
        .distance_metric = info.distance_metric,
        .seed = 42,
        .num_clusters =
            static_cast<uint32_t>(std::ceil(4.0 * std::sqrt(static_cast<double>(n_build)))),
        .normalize = true,
        .sampling_fraction = 1.0f,
        .kmeans_iters = 10,
        .hierarchical_indexing = false,
        .centroid_quantization = centroid_quantization,
    };
    PDX::PDXIndexU8 index(config);

    auto phases = OpenCsv(
        results_prefix + "_phases.csv",
        "dataset,workload,centroids,step,count,ms,per_second,clusters,index_bytes"
    );
    const auto log_phase = [&](const std::string& step, size_t count, double ms) {
        const double per_second = ms > 0 ? 1000.0 * static_cast<double>(count) / ms : 0.0;
        phases << dataset << "," << workload_name << "," << centroids_name << "," << step << ","
               << count << "," << ms << "," << per_second << "," << index.GetNumClusters() << ","
               << index.GetInMemorySizeInBytes() << "\n";
        phases.flush();
        std::cout << "phase " << step << " count=" << count << " ms=" << ms
                  << " per_s=" << per_second << " clusters=" << index.GetNumClusters()
                  << " bytes=" << index.GetInMemorySizeInBytes() << std::endl;
    };

    // Row bookkeeping, as in BenchmarkWorkload: deletes pop the latest row, appends reuse freed
    // rows first
    size_t next_row_id = 0;
    size_t next_data_index = 0;
    std::vector<std::pair<size_t, size_t>> live_entries; // (row_id, data_index)
    std::vector<size_t> freed_data_indices;
    std::vector<uint32_t> data_index_to_row_id(n);
    live_entries.reserve(n);
    for (size_t step_idx = 0; step_idx < workload.size(); ++step_idx) {
        const auto& step = workload[step_idx];
        const bool is_last_step = step_idx + 1 == workload.size();
        // The last append takes the rows left, so rounding the fractions cannot leave any out
        const size_t count = (is_last_step && step.type == StepType::INSERT)
                                 ? n - live_entries.size()
                                 : static_cast<size_t>(static_cast<double>(n) * step.proportion);
        const auto start = Clock::now();
        switch (step.type) {
        case StepType::BUILD:
            index.BuildIndex(data.data(), count);
            for (size_t i = 0; i < count; ++i) {
                live_entries.emplace_back(i, i);
                data_index_to_row_id[i] = static_cast<uint32_t>(i);
            }
            next_row_id = count;
            next_data_index = count;
            log_phase("build", count, ElapsedMs(start, Clock::now()));
            break;
        case StepType::INSERT:
            for (size_t i = 0; i < count; ++i) {
                size_t data_index;
                if (!freed_data_indices.empty()) {
                    data_index = freed_data_indices.back();
                    freed_data_indices.pop_back();
                } else {
                    data_index = next_data_index++;
                }
                const size_t row_id = next_row_id++;
                index.Append(row_id, data.data() + data_index * d);
                live_entries.emplace_back(row_id, data_index);
                data_index_to_row_id[data_index] = static_cast<uint32_t>(row_id);
            }
            log_phase("append" + std::to_string(step_idx), count, ElapsedMs(start, Clock::now()));
            break;
        case StepType::DELETE:
            for (size_t i = 0; i < count; ++i) {
                const auto [row_id, data_index] = live_entries.back();
                live_entries.pop_back();
                index.Delete(row_id);
                freed_data_indices.push_back(data_index);
            }
            log_phase("delete" + std::to_string(step_idx), count, ElapsedMs(start, Clock::now()));
            break;
        }
    }
    if (live_entries.size() != n) {
        std::cerr << "The workload ends with " << live_entries.size() << " rows, not " << n << "\n";
        return 1;
    }

    // The query rotation alone (normalize, then rotate), as the index transforms each query
    {
        PDX::ADSamplingPruner rotation(static_cast<uint32_t>(d), 42);
        std::vector<float> normalized(d);
        std::vector<float> rotated(d);
        std::vector<double> rotation_ms(n_queries);
        for (size_t q = 0; q < n_queries; ++q) {
            const auto start = Clock::now();
            const float* query = queries.data() + q * d;
            float squared_norm = 0.0f;
            for (size_t j = 0; j < d; ++j) {
                squared_norm += query[j] * query[j];
            }
            const float norm = std::sqrt(squared_norm);
            for (size_t j = 0; j < d; ++j) {
                normalized[j] = query[j] / norm;
            }
            rotation.PreprocessQuery(normalized.data(), rotated.data());
            rotation_ms[q] = ElapsedMs(start, Clock::now());
        }
        phases << dataset << "," << workload_name << "," << centroids_name
               << ",query_rotation_p50_ms," << n_queries << "," << Percentile(rotation_ms, 0.5)
               << ",0," << index.GetNumClusters() << "," << index.GetInMemorySizeInBytes() << "\n";
        phases.flush();
        std::cout << "query rotation p50 ms=" << Percentile(rotation_ms, 0.5) << std::endl;
    }

    const std::string gt_path =
        BenchmarkUtils::GROUND_TRUTH_DATA + info.pdx_dataset_name + "_100_norm";
    auto gt_buffer = MmapFile(gt_path);
    const auto* original_gt = reinterpret_cast<const uint32_t*>(gt_buffer.get());
    const size_t gt_k = BenchmarkUtils::GROUND_TRUTH_MAX_K;

    const auto run_query =
        [&](size_t q, uint32_t k, size_t nprobe, double& rank_ms, double& search_ms) {
            PDX::TopKHeap top_k_heap;
            const auto start = Clock::now();
            auto cursor =
                index.BeginIterativeSearch(queries.data() + q * d, k, top_k_heap, nullptr);
            const auto ranked = Clock::now();
            cursor->Next(nprobe);
            const auto searched = Clock::now();
            rank_ms = ElapsedMs(start, ranked);
            search_ms = ElapsedMs(ranked, searched);
            size_t true_positives = 0;
            std::unordered_set<uint32_t> expected;
            for (size_t j = 0; j < k; ++j) {
                expected.insert(data_index_to_row_id[original_gt[q * gt_k + j]]);
            }
            for (const auto& candidate : PDX::BuildResultSetFromHeap(k, top_k_heap.heap)) {
                true_positives += expected.count(candidate.index);
            }
            return static_cast<double>(true_positives) / k;
        };

    // Warm-up pass, untimed
    for (size_t q = 0; q < n_queries; ++q) {
        double rank_ms, search_ms;
        run_query(q, KS.back(), NPROBES.back(), rank_ms, search_ms);
    }

    auto search = OpenCsv(
        results_prefix + "_search.csv",
        "dataset,workload,centroids,nprobe,k,recall,total_p50_ms,total_p99_ms,total_mean_ms,"
        "rank_p50_ms,rank_p99_ms,rank_mean_ms,search_p50_ms,search_p99_ms,search_mean_ms"
    );
    for (size_t nprobe : NPROBES) {
        for (uint32_t k : KS) {
            std::vector<double> rank_ms(n_queries), search_ms(n_queries), total_ms(n_queries);
            double recall = 0.0;
            for (size_t q = 0; q < n_queries; ++q) {
                recall += run_query(q, k, nprobe, rank_ms[q], search_ms[q]);
                total_ms[q] = rank_ms[q] + search_ms[q];
            }
            recall /= static_cast<double>(n_queries);
            search << dataset << "," << workload_name << "," << centroids_name << "," << nprobe
                   << "," << k << "," << recall << "," << Percentile(total_ms, 0.5) << ","
                   << Percentile(total_ms, 0.99) << "," << Mean(total_ms) << ","
                   << Percentile(rank_ms, 0.5) << "," << Percentile(rank_ms, 0.99) << ","
                   << Mean(rank_ms) << "," << Percentile(search_ms, 0.5) << ","
                   << Percentile(search_ms, 0.99) << "," << Mean(search_ms) << "\n";
            search.flush();
            std::cout << "search nprobe=" << nprobe << " k=" << k << " recall=" << recall
                      << " p50_ms=" << Percentile(total_ms, 0.5)
                      << " p99_ms=" << Percentile(total_ms, 0.99)
                      << " rank_p50_ms=" << Percentile(rank_ms, 0.5)
                      << " search_p50_ms=" << Percentile(search_ms, 0.5) << std::endl;
        }
    }
    std::cout << "=== done" << std::endl;
    return 0;
}
