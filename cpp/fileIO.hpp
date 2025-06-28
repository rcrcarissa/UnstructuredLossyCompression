#pragma once

#include <iostream>
#include <fstream>
#include <map>
#include <cstdint>
#include <vector>

enum class DataType
{
    FLOAT,
    DOUBLE,
    SIZE_T,
    INT
};

void writeMap(const std::map<int, std::pair<uint32_t, int>> &m, const std::string &filename);

std::map<int, std::pair<uint32_t, int>> readMap(const std::string &filename);

void readRawArrayBinary(const std::string &fileName, void *data, std::size_t N, DataType type);

template <typename T>
void writeRawArrayBinary(const T *data, size_t N, const std::string &filename)
{
    static_assert(std::is_floating_point<T>::value, "Type must be float or double.");

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return;
    }

    // Write the data to the binary file
    file.write(reinterpret_cast<const char *>(data), N * sizeof(T));

    if (!file)
    {
        std::cerr << "Error: Failed to write to file." << std::endl;
    }

    file.close();
}

template <typename T>
std::vector<T> readVectorBinary(const std::string &filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    size_t size;
    ifs.read(reinterpret_cast<char *>(&size), sizeof(size));
    std::vector<T> vec(size);
    ifs.read(reinterpret_cast<char *>(vec.data()), sizeof(T) * size);
    return vec;
}

template <typename T>
void writeVectorBinary(const std::vector<T> &vec, const std::string &filename)
{
    std::ofstream ofs(filename, std::ios::binary);
    size_t size = vec.size();
    ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));
    ofs.write(reinterpret_cast<const char *>(vec.data()), sizeof(T) * size);
}
