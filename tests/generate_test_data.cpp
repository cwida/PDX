#include <fstream>
#include <iostream>
#include <vector>

#include "superkmeans/pdx/utils.h"

int main() {
    constexpr size_t N_TOTAL = 5500;
    constexpr size_t D = 384;
    constexpr size_t N_TRUE_CENTERS = 200;
    constexpr float CLUSTER_STD = 5.0f;
    constexpr float CENTER_SPREAD = 2.0f;
    constexpr unsigned int SEED = 42;

    std::cerr << "Generating test data (" << N_TOTAL << " x " << D << ")...\n";
    auto data =
        skmeans::MakeBlobs(N_TOTAL, D, N_TRUE_CENTERS, true, CLUSTER_STD, CENTER_SPREAD, SEED);

    std::string out_path = CMAKE_SOURCE_DIR "/tests/test_data.bin";
    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        std::cerr << "Error: Could not open " << out_path << " for writing\n";
        return 1;
    }
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    out.close();

    size_t size_mb = data.size() * sizeof(float) / 1024 / 1024;
    std::cerr << "Saved " << out_path << " (" << size_mb << " MB)\n";
    std::cerr << "First 5000 rows = train, last 500 = queries\n";
    return 0;
}
