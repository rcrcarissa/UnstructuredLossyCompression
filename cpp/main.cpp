#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <utility>
#include <chrono>
#include "../tools/eigen-3.4.0/Eigen/Dense"
// #include <Eigen/Dense>
#include "HuffmanCoder.cpp"
#include "fileIO.cpp"

constexpr size_t m = 16;

template <size_t dim>
struct cellFace
{
    // a cell face is an edge for triangular cells and a triangle for tetrahedral cells
    std::array<size_t, dim> nodes;

    cellFace() = default;

    cellFace(const std::array<size_t, dim> &input_nodes) : nodes(input_nodes)
    {
        std::sort(nodes.begin(), nodes.end());
    }

    bool operator==(const cellFace &f) const
    {
        return nodes == f.nodes;
    }
};

template <size_t dim>
struct cellFaceHash
{
    size_t operator()(const cellFace<dim> &f) const
    {
        size_t h = 0;
        for (size_t v : f.nodes)
        {
            h ^= std::hash<size_t>()(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

// custom hash for std::pair
namespace std
{
    template <>
    struct hash<std::pair<size_t, size_t>>
    {
        size_t operator()(const std::pair<size_t, size_t> &p) const noexcept
        {
            // A simple combination of the two elements
            return hash<size_t>()(p.first) ^ (hash<size_t>()(p.second) << 1);
        }
    };
}

template <typename T>
class RandomSet
{
    std::vector<T> data;
    std::unordered_map<T, size_t> index;

public:
    // default constructor
    RandomSet() = default;

    // constructor with initializer list
    RandomSet(std::initializer_list<T> init)
    {
        for (const T &val : init)
        {
            insert(val);
        }
    }

    // constructor from vector
    RandomSet(const std::vector<T> &values)
    {
        for (const T &val : values)
        {
            insert(val);
        }
    }

    // constructor filling with values 0 to n-1
    RandomSet(size_t n)
    {
        data.reserve(n);
        index.reserve(n);
        for (T i = 0; i < static_cast<T>(n); i++)
        {
            index[i] = data.size();
            data.push_back(i);
        }
    }

    bool insert(const T &val)
    {
        if (index.count(val))
            return false;
        index[val] = data.size();
        data.push_back(val);
        return true;
    }

    bool remove(const T &val)
    {
        if (!index.count(val))
            return false;
        size_t idx = index[val];
        T last = data.back();

        // move last element to the spot of the removed element
        data[idx] = last;
        index[last] = idx;

        data.pop_back();
        index.erase(val);
        return true;
    }

    const T &get(size_t i) const
    {
        return data[i];
    }

    T popRandom()
    {
        if (data.empty())
            throw std::out_of_range("Empty set");

        size_t r = rand() % data.size();
        T val = data[r];
        remove(val);
        return val;
    }

    T popFirst()
    {
        if (data.empty())
            throw std::out_of_range("Empty set");

        T val = data.front();
        remove(val);
        return val;
    }

    bool contains(const T &val) const
    {
        return index.count(val);
    }

    bool empty() const
    {
        return data.empty();
    }

    size_t size() const
    {
        return data.size();
    }
};

template <typename T>
T getRange(const T *arr, size_t N)
{
    T minVal = arr[0];
    T maxVal = arr[0];
    for (size_t i = 0; i < size; i++)
    {
        if (arr[i] < minVal)
            minVal = arr[i];
        if (arr[i] > maxVal)
            maxVal = arr[i];
    }
    return maxVal - minVal;
}

template <size_t dim>
void buildCellNeighbors(const size_t *cells, const size_t numCells, std::unordered_map<size_t, std::vector<size_t>> &cellNeighbors)
{
    std::unordered_map<cellFace<dim>, std::vector<size_t>, cellFaceHash<dim>> faceToCells;

    for (size_t cellId = 0; cellId < numCells; cellId++)
    {
        const size_t *cellNodes = &cells[cellId * (dim + 1)];

        for (size_t i = 0; i < dim + 1; i++)
        {
            std::array<size_t, dim> faceNodes;
            size_t idx = 0;
            for (size_t j = 0; j < dim + 1; j++)
            {
                if (j != i)
                {
                    faceNodes[idx++] = cellNodes[j];
                }
            }
            cellFace<dim> face(faceNodes);
            faceToCells[face].push_back(cellId);
        }
    }

    for (const auto &[face, cellList] : faceToCells)
    {
        if (cellList.size() > 1)
        {
            for (size_t i = 0; i < cellList.size(); i++)
            {
                for (size_t j = i + 1; j < cellList.size(); j++)
                {
                    size_t a = cellList[i];
                    size_t b = cellList[j];
                    cellNeighbors[a].push_back(b);
                    cellNeighbors[b].push_back(a);
                }
            }
        }
    }
}

template <size_t dim>
void buildNode2CellsMap(const size_t *cells, const size_t numCells, std::unordered_map<size_t, std::unordered_set<size_t>> &node2Cells)
{
    for (size_t i = 0; i < numCells; i++)
    {
        for (size_t j = i * (dim + 1); j < (i + 1) * (dim + 1); j++)
        {
            node2Cells[cells[j]].insert(i);
        }
    }
}

template <typename T, size_t dim>
T barycentricExtrap(const size_t nodeId, const size_t cellId, const T *nodeCoors, const size_t *cells, const T *values)
{
    size_t baseNodeId = cellId * (dim + 1);
    Eigen::Matrix<T, dim, dim> M;
    using RowVec = Eigen::Matrix<T, 1, dim, Eigen::RowMajor>;
    const RowVec base = Eigen::Map<const RowVec>(nodeCoors + cells[baseNodeId] * dim);

    for (size_t i = 0; i < dim; i++)
    {
        const RowVec cur = Eigen::Map<const RowVec>(nodeCoors + cells[baseNodeId + i + 1] * dim);
        M.row(i) = cur - base;
    }

    RowVec baryCoor = Eigen::Map<const RowVec>(nodeCoors + nodeId * dim);
    baryCoor -= base;
    baryCoor *= M.inverse();
    T lambda_0 = 1;
    for (size_t i = 0; i < dim; i++)
    {
        lambda_0 -= baryCoor(i);
    }
    T interp_result = values[cells[baseNodeId]] * lambda_0;
    for (size_t i = 0; i < dim; i++)
    {
        interp_result += values[cells[baseNodeId + i + 1]] * baryCoor(i);
    }
    return interp_result;
}

template <size_t dim>
void visitedUpdate(const size_t newlyVisitedNodeId, const size_t *cells, std::unordered_map<size_t, std::unordered_set<size_t>> node2Cells, bool *visitedCells, RandomSet<size_t> &unvisitedCells, bool *visitedNodes, size_t &numVisitedNodes)
{
    if (!visitedNodes[newlyVisitedNodeId])
    {
        visitedNodes[newlyVisitedNodeId] = true;
        numVisitedNodes++;
    }
    const auto &incidentCells = node2Cells[newlyVisitedNodeId];
    for (size_t cellId : incidentCells)
    {
        if (!visitedCells[cellId])
        {
            size_t startNodeId = cellId * (dim + 1);
            bool allNodeVisited = true;
            for (size_t i = 0; i < dim + 1; i++)
            {
                if (!visitedNodes[cells[startNodeId + i]])
                {
                    allNodeVisited = false;
                    break;
                }
            }
            if (allNodeVisited)
            {
                visitedCells[cellId] = true;
                unvisitedCells.remove(cellId);
            }
        }
    }
}

template <typename T, size_t dim>
void DFSCompression(const T xi, const T *nodeCoors, const size_t *cells, const T *values, const size_t numCells, const size_t numNodes, const size_t randomSeed, const std::string &fileName)
{
    bool visitedCells[numCells] = {};
    RandomSet<size_t> unvisitedCells(numCells);
    bool visitedNodes[numNodes] = {};
    size_t numVisitedNodes = 0;
    T decompValues[numNodes];
    std::unordered_map<size_t, std::unordered_set<size_t>> node2Cells;
    buildNode2CellsMap<dim>(cells, numCells, node2Cells);
    std::unordered_map<size_t, std::vector<size_t>> cellNeighbors;
    buildCellNeighbors<dim>(cells, numCells, cellNeighbors);
    std::vector<int> quantCodes;
    std::vector<T> losslessValues;
    quantCodes.reserve(numNodes);
    losslessValues.reserve(numNodes);
    srand(randomSeed);

    auto start_time = std::chrono::high_resolution_clock::now();

    while (numVisitedNodes < numNodes)
    {
        // pop a random seed cell from unvisited cells
        size_t seed = unvisitedCells.popRandom();
        visitedCells[seed] = true;

        // set all nodes of the seed as visited
        size_t startNodeId = seed * (dim + 1);
        for (size_t i = 0; i < dim + 1; i++)
        {
            size_t nodeId = cells[startNodeId + i];
            visitedUpdate<dim>(nodeId, cells, node2Cells, visitedCells, unvisitedCells, visitedNodes, numVisitedNodes);
            T orgValue = values[nodeId];
            decompValues[nodeId] = orgValue;
            losslessValues.push_back(orgValue);
        }

        // predict a sequence of cells starting from the seed
        DFSCompressOneSequence(xi, nodeCoors, cells, node2Cells, cellNeighbors, values, numNodes, seed, visitedCells, unvisitedCells, visitedNodes, numVisitedNodes, quantCodes, decompValues);
    }

    // lossless compression
    std::map<int, std::pair<uint32_t, int>> codeTable;
    auto compressed = huffmanZstdCompress(quantCodes, codeTable);

    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::seconds> duration = end_time - start_time;
    T compression_ratio = numNodes * sizeof(T) / (compressed.size() * 8 + losslessValues.size() * sizeof(T) + codeTable.size() * 96);
    T mse = 0;
    for (size_t i = 0; i < numNodes; i++)
    {
        T diff = values[i] - decompValues[i];
        mse += diff * diff;
    }
    mse /= numNodes;
    T rmse = sqrt(mse);
    T range = getRange(values, numNodes);
    T nrmse = rmse / range;
    T psnr = -20 * log10(nrmse);
    std::cout << "Compression time: " << duration.count() << " s" << std::endl;
    std::cout << "Compression ratio: " << compression_ratio << std::endl;
    std::cout << "MSE: " << mse << std::endl;
    std::cout << "RMSE: " << rmse << std::endl;
    std::cout << "NRMSE: " << nrmse << std::endl;
    std::cout << "PSNR (dB): " << psnr << std::endl;
    writeVectorBinary(compressed, fileName + ".comp");
    writeVectorBinary(losslessValues, fileName + ".lossless");
    writeMap(codeTable, fileName + ".codebook");
}

template <typename T, size_t dim>
void DFSCompressOneSequence(const T xi, const T *nodeCoors, const size_t *cells, const std::unordered_map<size_t, std::unordered_set<size_t>> &node2Cells, std::unordered_map<size_t, std::vector<size_t>> &cellNeighbors, const T *values, const size_t numNodes, const size_t seed, bool *visitedCells, RandomSet<size_t> &unvisitedCells, bool *visitedNodes, size_t &numVisitedNodes, std::vector<int> &quantCodes, T *decompValues)
{
    RandomSet<std::pair<size_t, size_t>> stack({{seed, numNodes}});
    while (stack.size() > 0)
    {
        auto [currCellId, prevCellId] = stack.popFirst();

        // determine & predict the newly introduced node
        if (!visitedCells[currCellId])
        {
            size_t currNodeId = currCellId * (dim + 1); // node to be updated
            size_t prevNodeId = prevCellId * (dim + 1);
            for (size_t i = 0; i < dim + 1; i++)
            {
                bool commonNode = false;
                for (size_t j = 0; j < dim + 1; j++)
                {
                    if (cells[currCellId + i] == cells[prevNodeId + j])
                    {
                        commonNode = true;
                        break;
                    }
                }
                if (!commonNode)
                {
                    currNodeId = cells[currCellId + i];
                    break;
                }
            }
            T predValue = barycentricExtrap<T, dim>(currNodeId, prevCellId, nodeCoors, cells, values);
            T dist = (values[currNodeId] - predValue) / (2 * xi);
            int quantCode = (1 << (m - 1)) + round(dist);
            if (quantCode < 1 || quantCode > 1 << m - 1)
            {
                break; // unpredictable; early stop
            }
            else
            {
                decompValues[currNodeId] = predValue + (quantCode - (1 << (m - 1))) * 2 * xi;
                quantCodes.push_back(quantCode);
                visitedUpdate<dim>(currNodeId, cells, node2Cells, visitedCells, unvisitedCells, visitedNodes, numVisitedNodes);
            }
        }
        std::vector<size_t> neighbors = cellNeighbors[currCellId];
        std::sort(neighbors.begin(), neighbors.end());
        for (size_t nextCellId : neighbors)
        {
            std::pair<size_t, size_t> cellPair = {nextCellId, currCellId};
            if (!visitedCells[nextCellId] && !stack.contains(cellPair))
            {
                stack.insert(cellPair);
            }
        }
    }
    quantCodes.push_back(1 << m);
}

template <typename T, size_t dim>
void DFSDecompression(const T xi, const T *nodeCoors, const size_t *cells, const size_t numCells, const size_t numNodes, const size_t randomSeed, const std::string &fileName, T *decompValues)
{
    bool visitedCells[numCells] = {};
    RandomSet<size_t> unvisitedCells(numCells);
    bool visitedNodes[numNodes] = {};
    size_t numVisitedNodes = 0;
    std::unordered_map<size_t, std::unordered_set<size_t>> node2Cells;
    buildNode2CellsMap<dim>(cells, numCells, node2Cells);
    std::unordered_map<size_t, std::vector<size_t>> cellNeighbors;
    buildCellNeighbors<dim>(cells, numCells, cellNeighbors);
    std::vector<uint8_t> compressed = readVectorBinary<uint8_t>(fileName + ".comp");
    std::vector<T> losslessValues = readVectorBinary<T>(fileName + ".lossless");
    std::map<int, std::pair<uint32_t, int>> codeTable = readMap(fileName + ".codebook");
    std::vector<int> quantCodes = huffmanZstdDecompress(compressed, codeTable, numNodes, compressed.size());
    size_t numSeeds = losslessValues.size() / (dim + 1);
    size_t losslessId = 0;
    srand(randomSeed);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < numSeeds; i++)
    {

        // pop a random seed cell from unvisited cells
        size_t seed = unvisitedCells.popRandom();
        visitedCells[seed] = true;

        // set all nodes of the seed as visited
        size_t startNodeId = seed * (dim + 1);
        for (size_t j = 0; j < dim + 1; j++)
        {
            size_t nodeId = cells[startNodeId + j];
            visitedUpdate<dim>(nodeId, cells, node2Cells, visitedCells, unvisitedCells, visitedNodes, numVisitedNodes);
            decompValues[nodeId] = losslessValues[losslessId];
            losslessId++;
        }

        // reconstruct a sequence of cells starting from the seed
        size_t quantCodeId = 0;
        DFSDecompressOneSequence(xi, nodeCoors, cells, node2Cells, cellNeighbors, numCells, numNodes, seed, visitedCells, unvisitedCells, visitedNodes, numVisitedNodes, quantCodes, quantCodeId, decompValues);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::seconds> duration = end_time - start_time;
    std::cout << "Decompression time: " << duration.count() << " s" << std::endl;
    writeRawArrayBinary(decompValues, fileName + ".decomp");
}

template <typename T, size_t dim>
void DFSDecompressOneSequence(const T xi, const T *nodeCoors, const size_t *cells, const std::unordered_map<size_t, std::unordered_set<size_t>> &node2Cells, std::unordered_map<size_t, std::vector<size_t>> &cellNeighbors, const size_t numCells, const size_t numNodes, const size_t seed, bool *visitedCells, RandomSet<size_t> &unvisitedCells, bool *visitedNodes, size_t &numVisitedNodes, const std::vector<int> &quantCodes, size_t &quantCodeId, T *decompValues)
{
    RandomSet<std::pair<size_t, size_t>> stack({{seed, numNodes}});
    int quantCodeMax = 1 << m;
    int quantCode = quantCodes[quantCodeId];
    while (quantCode < quantCodeMax)
    {
        auto [currCellId, prevCellId] = stack.popFirst();
        if (!visitedCells[currCellId])
        {
            size_t currNodeId = currCellId * (dim + 1); // node to be updated
            size_t prevNodeId = prevCellId * (dim + 1);
            for (size_t i = 0; i < dim + 1; i++)
            {
                bool commonNode = false;
                for (size_t j = 0; j < dim + 1; j++)
                {
                    if (cells[currCellId + i] == cells[prevNodeId + j])
                    {
                        commonNode = true;
                        break;
                    }
                }
                if (!commonNode)
                {
                    currNodeId = cells[currCellId + i];
                    break;
                }
            }
            T predValue = barycentricExtrap<T, dim>(currNodeId, prevCellId, nodeCoors, cells, decompValues);
            decompValues[currNodeId] = predValue + (quantCode - (1 << (m - 1))) * 2 * xi;
            visitedUpdate<dim>(currNodeId, cells, node2Cells, visitedCells, unvisitedCells, visitedNodes, numVisitedNodes);
            quantCodeId++;
            quantCode = quantCodes[quantCodeId];
        }
        std::vector<size_t> neighbors = cellNeighbors[currCellId];
        std::sort(neighbors.begin(), neighbors.end());
        for (size_t nextCellId : neighbors)
        {
            std::pair<size_t, size_t> cellPair = {nextCellId, currCellId};
            if (!visitedCells[nextCellId] && !stack.contains(cellPair))
            {
                stack.insert(cellPair);
            }
        }
    }
    quantCodeId++;
}

int main()
{
    size_t randomSeed = 34152;
    RandomSet<std::pair<size_t, size_t>> stack({{2, 1}});
    const auto &p = stack.get(0);
    printf("%zu, %zu\n", p.first, p.second);
    // // Triangular example (2 triangles)
    // const size_t triangleCells[] = {
    //     0, 3, 4, // cell 0
    //     0, 1, 4, // cell 1
    //     1, 4, 5,
    //     1, 2, 5,
    //     2, 5, 6,
    //     3, 4, 7,
    //     4, 7, 8,
    //     4, 5, 8,
    //     5, 8, 9,
    //     5, 6, 9};
    // std::unordered_map<size_t, std::unordered_set<size_t>> node2Cells;
    // buildNode2CellsMap<2>(triangleCells, 10, node2Cells);
    // for (const auto &[node, cells] : node2Cells)
    // {
    //     printf("%zu: ", node);
    //     for (size_t c : cells)
    //     {
    //         printf("%zu ", c);
    //     }
    //     printf("\n");
    // }
}
