#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <istream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>

inline std::unique_ptr<char[]> MmapFile(const std::string& filename) {
    std::ifstream input(filename, std::ios::binary | std::ios::ate);
    if (!input)
        throw std::runtime_error("Failed to open file");

    const auto file_size = static_cast<size_t>(input.tellg());
    input.seekg(0);

    std::unique_ptr<char[]> data(new char[file_size]);
    input.read(data.get(), static_cast<std::streamsize>(file_size));

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

    void Skip(size_t num_bytes) { ptr += num_bytes; }
};

struct StreamReader {
    std::istream& in;

    void Read(void* dst, size_t num_bytes) {
        in.read(static_cast<char*>(dst), static_cast<std::streamsize>(num_bytes));
        if (!in) {
            throw std::runtime_error("Unexpected end of a PDX index stream");
        }
    }

    void Skip(size_t num_bytes) {
        in.ignore(static_cast<std::streamsize>(num_bytes));
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
