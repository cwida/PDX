#ifndef PDX_TICTOC_HPP
#define PDX_TICTOC_HPP


#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <cstdio>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <limits>
#include <iomanip>
#include <chrono>
#include <unordered_map>
#include <filesystem>

/******************************************************************
 * Clock to benchmark algorithms runtime
 ******************************************************************/
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
};

#endif // PDX_TICTOC_HPP