#include <fstream>
#include <map>
#include <cstdint>
#include <vector>

void writeMap(const std::map<int, std::pair<uint32_t, int>> &m, const std::string &filename)
{
    std::ofstream ofs(filename, std::ios::binary);
    size_t size = m.size();
    ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));
    for (const auto &[key, val] : m)
    {
        ofs.write(reinterpret_cast<const char *>(&key), sizeof(key));
        ofs.write(reinterpret_cast<const char *>(&val.first), sizeof(val.first));
        ofs.write(reinterpret_cast<const char *>(&val.second), sizeof(val.second));
    }
}

std::map<int, std::pair<uint32_t, int>> readMap(const std::string &filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    std::map<int, std::pair<uint32_t, int>> m;
    size_t size;
    ifs.read(reinterpret_cast<char *>(&size), sizeof(size));
    for (size_t i = 0; i < size; ++i)
    {
        int key;
        uint32_t first;
        int second;
        ifs.read(reinterpret_cast<char *>(&key), sizeof(key));
        ifs.read(reinterpret_cast<char *>(&first), sizeof(first));
        ifs.read(reinterpret_cast<char *>(&second), sizeof(second));
        m[key] = {first, second};
    }
    return m;
}

template <typename T>
void writeVectorBinary(const std::vector<T> &vec, const std::string &filename)
{
    std::ofstream ofs(filename, std::ios::binary);
    size_t size = vec.size();
    ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));
    ofs.write(reinterpret_cast<const char *>(vec.data()), sizeof(T) * size);
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