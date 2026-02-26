#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "benchmark_utils.hpp"
#include "pdx/index.hpp"

template <typename IndexT>
void BuildAndSave(
    const RawDatasetInfo& info,
    const std::string& dataset,
    const std::string& index_type,
    const float* data
) {
    const size_t d = info.num_dimensions;
    const size_t n = info.num_embeddings;

    PDX::PDXIndexConfig index_config{
        .num_dimensions = static_cast<uint32_t>(d),
        .distance_metric = info.distance_metric,
        .seed = 42,
        .normalize = true,
        .sampling_fraction = 1.0f
    };

    std::cout << "Building " << index_type << " index...\n";
    auto build_start = std::chrono::high_resolution_clock::now();
    IndexT pdx_index(index_config);
    pdx_index.BuildIndex(data, n);
    auto build_end = std::chrono::high_resolution_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
    std::cout << "Build time: " << build_ms << " ms\n";
    std::cout << "Clusters: " << pdx_index.GetNumClusters() << "\n";
    std::cout << "Index in-memory size: " << std::fixed << std::setprecision(2)
              << static_cast<double>(pdx_index.GetInMemorySizeInBytes()) / (1024.0 * 1024.0)
              << " MB\n";

    std::string save_path = BenchmarkUtils::PDX_DATA + dataset + "-" + index_type;
    std::cout << "Saving to " << save_path << "...\n";
    pdx_index.Save(save_path);
    std::cout << "Done.\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset> [index_type]\n";
        std::cerr << "Index types: pdx_f32 (default), pdx_u8, pdx_tree_f32, pdx_tree_u8\n";
        std::cerr << "Available datasets:";
        for (const auto& [name, _] : RAW_DATASET_PARAMS) {
            std::cerr << " " << name;
        }
        std::cerr << "\n";
        return 1;
    }
    std::string dataset = argv[1];
    std::string index_type = (argc > 2) ? argv[2] : "pdx_f32";

    auto it = RAW_DATASET_PARAMS.find(dataset);
    if (it == RAW_DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset: " << dataset << "\n";
        return 1;
    }
    const auto& info = it->second;
    const size_t n = info.num_embeddings;
    const size_t d = info.num_dimensions;

    std::cout << "==> PDX Serialization\n";
    std::cout << "Dataset: " << dataset << " (" << info.pdx_dataset_name << ", n=" << n
              << ", d=" << d << ")\n";
    std::cout << "Index type: " << index_type << "\n";

    // Read raw data
    std::string data_path = RAW_DATA_DIR + "/data_" + dataset + ".bin";
    std::vector<float> data(n * d);
    {
        std::ifstream file(data_path, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open " << data_path << "\n";
            return 1;
        }
        file.read(reinterpret_cast<char*>(data.data()), n * d * sizeof(float));
    }

    // Ensure output directory exists
    std::filesystem::create_directories(BenchmarkUtils::PDX_DATA);

    std::string save_path = BenchmarkUtils::PDX_DATA + dataset + "-" + index_type;

    if (index_type == "pdx_f32") {
        BuildAndSave<PDX::PDXIndexF32>(info, dataset, index_type, data.data());
    } else if (index_type == "pdx_u8") {
        BuildAndSave<PDX::PDXIndexU8>(info, dataset, index_type, data.data());
    } else if (index_type == "pdx_tree_f32") {
        BuildAndSave<PDX::PDXTreeIndexF32>(info, dataset, index_type, data.data());
    } else if (index_type == "pdx_tree_u8") {
        BuildAndSave<PDX::PDXTreeIndexU8>(info, dataset, index_type, data.data());
    } else {
        std::cerr << "Unknown index type: " << index_type << "\n";
        std::cerr << "Valid types: pdx_f32, pdx_u8, pdx_tree_f32, pdx_tree_u8\n";
        return 1;
    }

    // Verify: load back without knowing the type and run queries
    std::cout << "\n==> Verification: Loading index from " << save_path << "...\n";
    auto loaded_index = PDX::LoadPDXIndex(save_path);
    std::cout << "Loaded index in-memory size: " << std::fixed << std::setprecision(2)
              << static_cast<double>(loaded_index->GetInMemorySizeInBytes()) / (1024.0 * 1024.0)
              << " MB\n";

    // Load queries
    std::unique_ptr<char[]> query_ptr =
        MmapFile(BenchmarkUtils::QUERIES_DATA + info.pdx_dataset_name);
    auto* queries = reinterpret_cast<float*>(query_ptr.get());
    queries += 1; // skip header

    // Load ground truth
    std::unique_ptr<char[]> gt_buffer =
        MmapFile(BenchmarkUtils::GROUND_TRUTH_DATA + info.pdx_dataset_name + "_100_norm");
    auto* int_ground_truth = reinterpret_cast<uint32_t*>(gt_buffer.get());

    const size_t n_queries = info.num_queries;
    const uint8_t KNN = BenchmarkUtils::KNN;

    loaded_index->SetNProbe(25);
    float recalls = 0;
    for (size_t l = 0; l < n_queries; ++l) {
        auto result = loaded_index->Search(queries + l * d, KNN);
        BenchmarkUtils::VerifyResult<true>(recalls, result, KNN, int_ground_truth, l);
    }
    float avg_recall = recalls / n_queries;
    std::cout << "Recall@" << +KNN << " (nprobe=25): " << avg_recall << "\n";

    std::cout << "Verification complete.\n";
    return 0;
}
