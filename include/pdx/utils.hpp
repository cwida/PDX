#pragma once

#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <istream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef linux
#include <linux/mman.h>
#endif

inline std::unique_ptr<char[]> MmapFile(const std::string& filename) {
    struct stat file_stats {};
    int fd = ::open(filename.c_str(), O_RDONLY);
    if (fd == -1)
        throw std::runtime_error("Failed to open file");

    fstat(fd, &file_stats);
    size_t file_size = file_stats.st_size;

    std::unique_ptr<char[]> data(new char[file_size]);
    std::ifstream input(filename, std::ios::binary);
    input.read(data.get(), file_size);

    return data;
}

namespace PDX {

// The Load functions read the format through one of these, so it is implemented once for buffers
// and for streams.
struct BufferReader {
    char*& ptr;

    void Read(void* dst, size_t num_bytes) {
        std::memcpy(dst, ptr, num_bytes);
        ptr += num_bytes;
    }
};

struct StreamReader {
    std::istream& in;

    void Read(void* dst, size_t num_bytes) {
        in.read(static_cast<char*>(dst), static_cast<std::streamsize>(num_bytes));
        if (!in) {
            throw std::runtime_error("Unexpected end of a PDX index stream");
        }
    }
};

template <class T, class Reader>
T ReadValue(Reader& reader) {
    T value;
    reader.Read(&value, sizeof(T));
    return value;
}

template <class T>
void WriteValue(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

} // namespace PDX
