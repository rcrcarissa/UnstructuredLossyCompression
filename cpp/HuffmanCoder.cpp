#include <iostream>
#include <vector>
#include <map>
#include <queue>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include <zstd.h>

template <typename T>
struct HuffmanNode
{
    T value;
    size_t freq;
    std::shared_ptr<HuffmanNode<T>> left, right;

    HuffmanNode(T val, size_t f) : value(val), freq(f) {}
    HuffmanNode(std::shared_ptr<HuffmanNode<T>> l, std::shared_ptr<HuffmanNode<T>> r)
        : value(), freq(l->freq + r->freq), left(l), right(r) {}

    bool is_leaf() const { return !left && !right; }
};

template <typename T>
struct Compare
{
    bool operator()(const std::shared_ptr<HuffmanNode<T>> &a,
                    const std::shared_ptr<HuffmanNode<T>> &b)
    {
        return a->freq > b->freq;
    }
};

template <typename T>
void buildCodeTable(const std::shared_ptr<HuffmanNode<T>> &node,
                    std::map<T, std::pair<uint32_t, int>> &table,
                    uint32_t code = 0, int length = 0)
{
    if (!node)
        return;
    if (node->is_leaf())
    {
        table[node->value] = {code, length};
    }
    else
    {
        buildCodeTable(node->left, table, code << 1, length + 1);
        buildCodeTable(node->right, table, (code << 1) | 1, length + 1);
    }
}

template <typename T>
std::shared_ptr<HuffmanNode<T>> rebuildTreeFromCodeMap(const std::map<T, std::pair<uint32_t, int>> &codeMap)
{
    auto root = std::make_shared<HuffmanNode<T>>(T{}, 0);
    for (const auto &[sym, code_len] : codeMap)
    {
        uint32_t code = code_len.first;
        int len = code_len.second;
        auto node = root;
        for (int i = len - 1; i >= 0; --i)
        {
            bool bit = (code >> i) & 1;
            if (bit)
            {
                if (!node->right)
                    node->right = std::make_shared<HuffmanNode<T>>(T{}, 0);
                node = node->right;
            }
            else
            {
                if (!node->left)
                    node->left = std::make_shared<HuffmanNode<T>>(T{}, 0);
                node = node->left;
            }
        }
        node->value = sym;
    }
    return root;
}

template <typename T>
std::shared_ptr<HuffmanNode<T>> buildTree(const std::map<T, size_t> &freq)
{
    std::priority_queue<std::shared_ptr<HuffmanNode<T>>,
                        std::vector<std::shared_ptr<HuffmanNode<T>>>, Compare<T>>
        pq;
    for (const auto &[sym, f] : freq)
    {
        pq.push(std::make_shared<HuffmanNode<T>>(sym, f));
    }
    while (pq.size() > 1)
    {
        auto a = pq.top();
        pq.pop();
        auto b = pq.top();
        pq.pop();
        pq.push(std::make_shared<HuffmanNode<T>>(a, b));
    }
    return pq.top();
}

class BitWriter
{
    std::vector<uint8_t> buffer;
    uint8_t currentByte = 0;
    int bitPos = 0;

public:
    void writeBits(uint32_t code, int length)
    {
        for (int i = length - 1; i >= 0; --i)
        {
            currentByte |= ((code >> i) & 1) << (7 - bitPos++);
            if (bitPos == 8)
            {
                buffer.push_back(currentByte);
                currentByte = 0;
                bitPos = 0;
            }
        }
    }

    std::vector<uint8_t> finalize()
    {
        if (bitPos > 0)
            buffer.push_back(currentByte);
        return buffer;
    }
};

class BitReader
{
    const std::vector<uint8_t> &buffer;
    size_t byteIndex = 0;
    int bitIndex = 0;

public:
    BitReader(const std::vector<uint8_t> &buf) : buffer(buf) {}

    // Read next bit; returns -1 if out of bounds
    int readBit()
    {
        if (byteIndex >= buffer.size())
            return -1;
        int bit = (buffer[byteIndex] >> (7 - bitIndex)) & 1;
        if (++bitIndex == 8)
        {
            bitIndex = 0;
            ++byteIndex;
        }
        return bit;
    }
};

template <typename T>
std::vector<uint8_t> huffmanZstdCompress(const std::vector<T> &data, std::map<T, std::pair<uint32_t, int>> &codeTableOut, int zstdLevel = 3)
{
    // Frequency count
    std::map<T, size_t> freq;
    for (const auto &val : data)
        freq[val]++;

    // Build Huffman tree & code table
    auto root = buildTree(freq);
    buildCodeTable(root, codeTableOut);

    // Bit-pack using code table
    BitWriter writer;
    for (const auto &val : data)
    {
        const auto &[code, len] = codeTableOut[val];
        writer.writeBits(code, len);
    }
    auto bitPacked = writer.finalize();

    // Compress with ZSTD
    size_t bound = ZSTD_compressBound(bitPacked.size());
    std::vector<uint8_t> compressed(bound);
    size_t cSize = ZSTD_compress(compressed.data(), bound, bitPacked.data(), bitPacked.size(), zstdLevel);
    if (ZSTD_isError(cSize))
        throw std::runtime_error(ZSTD_getErrorName(cSize));
    compressed.resize(cSize);
    return compressed;
}

template <typename T>
std::vector<T> huffmanZstdDecompress(const std::vector<uint8_t> &compressed, const std::map<T, std::pair<uint32_t, int>> &codeTable, size_t numElements, size_t originalBitstreamSize)
{
    // ZSTD decompress into bit-packed buffer
    std::vector<uint8_t> decompressed(originalBitstreamSize);
    size_t res = ZSTD_decompress(decompressed.data(), originalBitstreamSize, compressed.data(), compressed.size());
    if (ZSTD_isError(res))
        throw std::runtime_error(ZSTD_getErrorName(res));

    // Rebuild tree
    auto root = rebuildTreeFromCodeMap(codeTable);

    // Decode
    BitReader reader(decompressed);
    std::vector<T> output;
    output.reserve(numElements);
    auto node = root;

    while (output.size() < numElements)
    {
        int bit = reader.readBit();
        if (bit == -1)
            throw std::runtime_error("Unexpected end of bitstream");

        node = bit ? node->right : node->left;
        if (node->is_leaf())
        {
            output.push_back(node->value);
            node = root;
        }
    }

    return output;
}

// int main()
// {
//     std::vector<int> data = {3, 3, 4, 2, 1, 1, 1, 3, 1, 1, 1, 3, 2, 1, 2, 4, 2, 1, 1, 3};

//     std::map<int, std::pair<uint32_t, int>> codeTable;
//     auto compressed = huffmanZstdCompress(data, codeTable);

//     std::cout << "Compressed size: " << compressed.size() << " bytes\n";
//     std::cout << "Code Table:\n";
//     for (const auto &[val, codeInfo] : codeTable)
//     {
//         std::cout << val << " => bits: ";
//         for (int i = codeInfo.second - 1; i >= 0; --i)
//             std::cout << ((codeInfo.first >> i) & 1);
//         std::cout << "\n";
//     }

//     size_t originalDataSize = data.size();
//     size_t compressedDataSize = compressed.size();
//     auto decompressed = huffmanZstdDecompress(compressed, codeTable, originalDataSize, compressedDataSize);
//     for (auto v : decompressed)
//     {
//         std::cout << v << " ";
//     }
//     std::cout << std::endl;
// }
