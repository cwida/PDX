#ifndef PDX_UTILS_HPP
#define PDX_UTILS_HPP

#include <cstdint>
#include <memory>
#include <fcntl.h>
#include <fstream>
#include <cstdio>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef linux
#include <linux/mman.h>
#endif

/******************************************************************
 * File reader
 ******************************************************************/
inline std::unique_ptr<char[]> MmapFile(const std::string& filename) {
    struct stat file_stats {};
    int fd = ::open(filename.c_str(), O_RDONLY);
    if (fd == -1) throw std::runtime_error("Failed to open file");

    fstat(fd, &file_stats);
    size_t file_size = file_stats.st_size;

    auto data = std::make_unique<char[]>(file_size);
    std::ifstream input(filename, std::ios::binary);
    input.read(data.get(), file_size);

    return data;
}

#endif //PDX_UTILS_HPP
