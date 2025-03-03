#ifndef EMBEDDINGSEARCH_UTILS_HPP
#define EMBEDDINGSEARCH_UTILS_HPP

#include <cstdint>
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
 * File readers (nothing fancy)
 ******************************************************************/
float* MmapFile32(size_t& n_values, const std::string& filename) {
    size_t file_size;
    struct stat file_stats;
    
    int fd = ::open(filename.c_str(), O_RDONLY);
    fstat(fd, &file_stats);
    file_size = file_stats.st_size;
    float * file_pointer = new float[file_size / sizeof(float)];
    std::ifstream input(filename.c_str(), std::ios::binary);
    input.read((char*) file_pointer, file_size);
    n_values = file_size / sizeof(float);
    input.close();
    return file_pointer;
}

float* MmapFile32(const std::string& filename) {
    size_t file_size;
    struct stat file_stats;

    int fd = ::open(filename.c_str(), O_RDONLY);
    fstat(fd, &file_stats);
    file_size = file_stats.st_size;
    float * file_pointer = new float[file_size/sizeof(float)];
    std::ifstream input(filename.c_str(), std::ios::binary);
    input.read((char*) file_pointer, file_size);
    input.close();
    return file_pointer;
}


#endif //EMBEDDINGSEARCH_UTILS_HPP
