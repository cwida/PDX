#pragma once

#include <cstdint>
#include <memory>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <chrono>

#ifdef linux
#include <linux/mman.h>
#endif

class TicToc {
public:
    size_t accum_time = 0;
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    void Reset() {
        accum_time = 0;
        start = std::chrono::high_resolution_clock::now();
    }

    inline void Tic() {
        start = std::chrono::high_resolution_clock::now();
    }

    inline void Toc() {
        auto end = std::chrono::high_resolution_clock::now();
        accum_time += std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - start).count();
    }

    double GetMilliseconds() const {
        return static_cast<double>(accum_time) / 1e6;
    }
};

inline std::unique_ptr<char[]> MmapFile(const std::string& filename) {
    struct stat file_stats {};
    int fd = ::open(filename.c_str(), O_RDONLY);
    if (fd == -1) throw std::runtime_error("Failed to open file");

    fstat(fd, &file_stats);
    size_t file_size = file_stats.st_size;

    std::unique_ptr<char[]> data(new char[file_size]);
    std::ifstream input(filename, std::ios::binary);
    input.read(data.get(), file_size);

    return data;
}
