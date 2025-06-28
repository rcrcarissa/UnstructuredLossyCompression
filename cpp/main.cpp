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
#include <cstring>
#include "HuffmanCoder.hpp"
#include "fileIO.hpp"

constexpr size_t m = 16;
constexpr size_t randomSeed = 7548921;
std::string nodesCoorFile;
std::string cellsFile;
std::string orgDataFile = "";
std::string compressedFile;
std::string decompressedFile = "";
size_t numNodes;
size_t numCells;
size_t face_dim;
size_t node_dim;
bool isDouble = false;
bool isABS = false;
float err_bound;

template <size_t face_dim>
struct cellFace
{
    // a cell face is an edge for triangular cells and a triangle for tetrahedral cells
    std::array<size_t, face_dim> nodes;

    cellFace() = default;

    cellFace(const std::array<size_t, face_dim> &input_nodes) : nodes(input_nodes) {}

    bool operator==(const cellFace &f) const
    {
        return nodes == f.nodes;
    }
};

template <size_t face_dim>
struct cellFaceHash
{
    size_t operator()(const cellFace<face_dim> &f) const
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

    T popBack()
    {
        if (data.empty())
            throw std::out_of_range("Empty set");

        T val = data.back();
        data.pop_back();
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
T getRange(const T *arr, size_t N);

template <size_t face_dim>
void buildCellNeighbors(const size_t *cells, const size_t numCells, std::vector<std::vector<size_t>> &cellNeighbors);

template <size_t face_dim>
void buildNode2CellsMap(const size_t *cells, const size_t numCells, std::unordered_map<size_t, std::unordered_set<size_t>> &node2Cells);

template <typename T, size_t face_dim, size_t node_dim>
T barycentricExtrap(const size_t nodeId, const size_t cellId, const T *nodeCoors, const size_t *cells, const T *values);

template <size_t face_dim>
void visitedUpdate(const size_t newlyVisitedNodeId, const size_t *cells, const std::unordered_map<size_t, std::unordered_set<size_t>> &node2Cells, bool *visitedCells, RandomSet<size_t> &unvisitedCells, bool *visitedNodes, size_t &numVisitedNodes);

template <typename T, size_t face_dim, size_t node_dim>
void DFSCompression(const T xi, const T *nodeCoors, const size_t *cells, const T *values, const size_t numCells, const size_t numNodes, const size_t randomSeed, const std::string &fileName);

template <typename T, size_t face_dim, size_t node_dim>
void DFSCompressOneSequence(const T xi, const T *nodeCoors, const size_t *cells, const std::unordered_map<size_t, std::unordered_set<size_t>> &node2Cells, std::vector<std::vector<size_t>> &cellNeighbors, const T *values, const size_t numNodes, const size_t seed, bool *visitedCells, RandomSet<size_t> &unvisitedCells, bool *visitedNodes, size_t &numVisitedNodes, std::vector<int> &quantCodes, T *decompValues);

template <typename T, size_t face_dim, size_t node_dim>
void DFSDecompression(const T xi, const T *nodeCoors, const size_t *cells, const size_t numCells, const size_t numNodes, const size_t randomSeed, const std::string &fileName);

template <typename T, size_t face_dim, size_t node_dim>
void DFSDecompressOneSequence(const T xi, const T *nodeCoors, const size_t *cells, const std::unordered_map<size_t, std::unordered_set<size_t>> &node2Cells, std::vector<std::vector<size_t>> &cellNeighbors, const size_t numCells, const size_t numNodes, const size_t seed, bool *visitedCells, RandomSet<size_t> &unvisitedCells, bool *visitedNodes, size_t &numVisitedNodes, const std::vector<int> &quantCodes, size_t &quantCodeId, T *decompValues);

void parseError(const char error[])
{
    std::cout << error << std::endl;
    std::cout << "Usage:\n";
    std::cout << "  -iv <file_path> : Specify the input file of coordinates of vertices\n";
    std::cout << "  -ic <file_path> : Specify the input file of cells\n";
    std::cout << "  -id <file_path> : Specify the input file of data on vertices\n";
    std::cout << "  -z <file_path>  : Specify the compressed file\n";
    std::cout << "  -o <file_path>  : Specify the decompressed file (optional)\n";
    std::cout << "  -2 <nv>         : <nv> vertices in 2D\n";
    std::cout << "  -3 <nv>         : <nv> vertices in 3D\n";
    std::cout << "  -tri <nc>       : <nc> triangular cells\n";
    std::cout << "  -tet <nc>       : <nc> tetrahedral cells\n";
    std::cout << "  -f              : Use float data type\n";
    std::cout << "  -d              : Use double data type\n";
    std::cout << "  -M ABS <xi>     : Specify the absolute error bound\n";
    std::cout << "  -M REL <xi>     : Specify the relative error bound\n";
    exit(EXIT_FAILURE);
}

void Parsing(int argc, char *argv[])
{
    int inputFileSpecified = 0;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "-iv")
        {
            if (i + 1 >= argc)
                parseError("Missing input file of coordinates of vertices");
            nodesCoorFile = argv[++i];
            inputFileSpecified++;
        }
        else if (arg == "-ic")
        {
            if (i + 1 >= argc)
                parseError("Missing input file of cells");
            cellsFile = argv[++i];
            inputFileSpecified++;
        }
        else if (arg == "-id")
        {
            orgDataFile = argv[++i];
            inputFileSpecified++;
        }
        else if (arg == "-z")
        {
            if (i + 1 >= argc)
                parseError("Missing name of compressed file");
            compressedFile = argv[++i];
        }
        else if (arg == "-o")
        {
            decompressedFile = argv[++i];
        }
        else if (arg == "-2" || arg == "-3")
        {
            if (arg == "-2")
            {
                node_dim = 2;
            }
            else
            {
                node_dim = 3;
            }
            numNodes = std::stoi(argv[++i]);
            if (i + 1 >= argc)
            {
                parseError("Missing number of vertices");
            }
        }
        else if (arg == "-tri" || arg == "-tet")
        {
            if (arg == "-tri")
            {
                face_dim = 2;
            }
            else
            {
                face_dim = 3;
            }
            numCells = std::stoi(argv[++i]);
            if (i + 1 >= argc)
            {
                parseError("Missing number of cells");
            }
        }
        else if (arg == "-f")
        {
            isDouble = false; // Use float type
        }
        else if (arg == "-d")
        {
            isDouble = true; // Use double type
        }
        else if (arg == "-M")
        {
            isABS = std::strcmp(argv[++i], "ABS") == 0;
            if (i + 1 >= argc)
                parseError("Missing relative error bound");
            err_bound = std::stof(argv[++i]);
        }
        else
        {
            parseError("Unknown argument");
        }
    }

    if (inputFileSpecified < 3)
    {
        parseError("Input files (-iv, -ic, one of -id and -z) are mandatory");
    }
}

template <typename T>
T getRange(const T *arr, size_t N)
{
    T minVal = arr[0];
    T maxVal = arr[0];
    for (size_t i = 0; i < N; i++)
    {
        if (arr[i] < minVal)
            minVal = arr[i];
        if (arr[i] > maxVal)
            maxVal = arr[i];
    }
    return maxVal - minVal;
}

template <size_t face_dim>
void buildCellNeighbors(const size_t *cells, const size_t numCells, std::vector<std::vector<size_t>> &cellNeighbors)
{
    std::unordered_map<cellFace<face_dim>, std::vector<size_t>, cellFaceHash<face_dim>> faceToCells;
    size_t cellNodes[face_dim + 1];

    for (size_t cellId = 0; cellId < numCells; cellId++)
    {
        std::memcpy(cellNodes, cells + cellId * (face_dim + 1), (face_dim + 1) * sizeof(size_t));
        std::sort(cellNodes, cellNodes + face_dim + 1);

        for (size_t i = 0; i < face_dim + 1; i++)
        {
            std::array<size_t, face_dim> faceNodes;
            size_t idx = 0;
            for (size_t j = 0; j < face_dim + 1; j++)
            {
                if (j != i)
                {
                    faceNodes[idx++] = cellNodes[j];
                }
            }
            cellFace<face_dim> face(faceNodes);
            faceToCells[face].push_back(cellId);
        }
    }

    cellNeighbors.resize(numCells);
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

template <size_t face_dim>
void buildNode2CellsMap(const size_t *cells, const size_t numCells, std::unordered_map<size_t, std::unordered_set<size_t>> &node2Cells)
{
    for (size_t i = 0; i < numCells; i++)
    {
        for (size_t j = i * (face_dim + 1); j < (i + 1) * (face_dim + 1); j++)
        {
            node2Cells[cells[j]].insert(i);
        }
    }
}

template <typename T, size_t face_dim, size_t node_dim>
T barycentricExtrap(const size_t nodeId, const size_t cellId, const T *nodeCoors, const size_t *cells, const T *values)
{
    // fetch node coordinates
    const T *p = &nodeCoors[nodeId * node_dim];

    // fetch simplex node indices
    const size_t *simplex = &cells[cellId * (face_dim + 1)];
    const T *v0 = &nodeCoors[simplex[0] * node_dim];
    const T *v1 = &nodeCoors[simplex[1] * node_dim];
    const T *v2 = &nodeCoors[simplex[2] * node_dim];

    T w0, w1, w2, w3 = 0;

    if constexpr (face_dim == 2 && node_dim == 2)
    {
        // triangle in 2D
        T x = p[0], y = p[1];
        T x0 = v0[0], y0 = v0[1];
        T x1 = v1[0], y1 = v1[1];
        T x2 = v2[0], y2 = v2[1];

        T detT = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
        w0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / detT;
        w1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / detT;
        w2 = 1 - w0 - w1;

        return w0 * values[simplex[0]] + w1 * values[simplex[1]] + w2 * values[simplex[2]];
    }

    else if constexpr (face_dim == 2 && node_dim == 3)
    {
        // triangle in 3D (project to 2D plane)
        const T *v = p;

        // edges
        T e0[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
        T e1[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
        T vp[3] = {v[0] - v0[0], v[1] - v0[1], v[2] - v0[2]};

        // cross products
        T n[3] = {
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0]};
        T area = 0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);

        // areas of sub-triangles
        T a0[3] = {
            (v1[1] - v[1]) * (v2[2] - v[2]) - (v1[2] - v[2]) * (v2[1] - v[1]),
            (v1[2] - v[2]) * (v2[0] - v[0]) - (v1[0] - v[0]) * (v2[2] - v[2]),
            (v1[0] - v[0]) * (v2[1] - v[1]) - (v1[1] - v[1]) * (v2[0] - v[0])};
        T a1[3] = {
            (v2[1] - v[1]) * (v0[2] - v[2]) - (v2[2] - v[2]) * (v0[1] - v[1]),
            (v2[2] - v[2]) * (v0[0] - v[0]) - (v2[0] - v[0]) * (v0[2] - v[2]),
            (v2[0] - v[0]) * (v0[1] - v[1]) - (v2[1] - v[1]) * (v0[0] - v[0])};

        T denom = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
        w0 = ((n[0] * a0[0] + n[1] * a0[1] + n[2] * a0[2]) / denom);
        w1 = ((n[0] * a1[0] + n[1] * a1[1] + n[2] * a1[2]) / denom);
        w2 = 1 - w0 - w1;

        return w0 * values[simplex[0]] + w1 * values[simplex[1]] + w2 * values[simplex[2]];
    }

    else if constexpr (face_dim == 3 && node_dim == 3)
    {
        // tetrahedron in 3D
        const T *v3 = &nodeCoors[simplex[3] * node_dim];
        const T *v = p;

        auto det = [](const T *a, const T *b, const T *c) -> T
        {
            return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
        };

        T a[3] = {v0[0] - v3[0], v0[1] - v3[1], v0[2] - v3[2]};
        T b[3] = {v1[0] - v3[0], v1[1] - v3[1], v1[2] - v3[2]};
        T c[3] = {v2[0] - v3[0], v2[1] - v3[1], v2[2] - v3[2]};
        T d[3] = {v[0] - v3[0], v[1] - v3[1], v[2] - v3[2]};

        T detT = det(a, b, c);
        w0 = det(d, b, c) / detT;
        w1 = det(a, d, c) / detT;
        w2 = det(a, b, d) / detT;
        w3 = 1 - w0 - w1 - w2;

        return w0 * values[simplex[0]] + w1 * values[simplex[1]] + w2 * values[simplex[2]] + w3 * values[simplex[3]];
    }
}

template <size_t face_dim>
void visitedUpdate(const size_t newlyVisitedNodeId, const size_t *cells, const std::unordered_map<size_t, std::unordered_set<size_t>> &node2Cells, bool *visitedCells, RandomSet<size_t> &unvisitedCells, bool *visitedNodes, size_t &numVisitedNodes)
{
    if (!visitedNodes[newlyVisitedNodeId])
    {
        visitedNodes[newlyVisitedNodeId] = true;
        numVisitedNodes++;
    }

    auto it = node2Cells.find(newlyVisitedNodeId);
    if (it == node2Cells.end())
        return; // key doesn't exist
    const auto &incidentCells = it->second;

    for (size_t cellId : incidentCells)
    {
        if (!visitedCells[cellId])
        {
            size_t startNodeId = cellId * (face_dim + 1);
            bool allNodeVisited = true;
            for (size_t i = 0; i < face_dim + 1; i++)
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

template <typename T, size_t face_dim, size_t node_dim>
void DFSCompression(const T xi, const T *nodeCoors, const size_t *cells, const T *values, const size_t numCells, const size_t numNodes, const size_t randomSeed, const std::string &fileName)
{
    bool visitedCells[numCells] = {};
    RandomSet<size_t> unvisitedCells(numCells);
    bool visitedNodes[numNodes] = {};
    size_t numVisitedNodes = 0;
    T *decompValues = new T[numNodes];
    std::unordered_map<size_t, std::unordered_set<size_t>> node2Cells;
    buildNode2CellsMap<face_dim>(cells, numCells, node2Cells);
    std::vector<std::vector<size_t>> cellNeighbors;
    buildCellNeighbors<face_dim>(cells, numCells, cellNeighbors);
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
        size_t startNodeId = seed * (face_dim + 1);
        for (size_t i = 0; i < face_dim + 1; i++)
        {
            size_t nodeId = cells[startNodeId + i];
            visitedUpdate<face_dim>(nodeId, cells, node2Cells, visitedCells, unvisitedCells, visitedNodes, numVisitedNodes);
            T orgValue = values[nodeId];
            decompValues[nodeId] = orgValue;
            losslessValues.push_back(orgValue);
        }

        // predict a sequence of cells starting from the seed
        DFSCompressOneSequence<T, face_dim, node_dim>(xi, nodeCoors, cells, node2Cells, cellNeighbors, values, numNodes, seed, visitedCells, unvisitedCells, visitedNodes, numVisitedNodes, quantCodes, decompValues);
    }

    // lossless compression
    std::map<int, std::pair<uint32_t, int>> codeTable;
    auto compressed = huffmanZstdCompress(quantCodes, codeTable);

    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    T compression_ratio = static_cast<T>(numNodes * sizeof(T)) / (compressed.size() + losslessValues.size() * sizeof(T));
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
    std::cout << "Compression time: " << duration.count() << " ms" << std::endl;
    std::cout << "Compression ratio: " << compression_ratio << std::endl;
    std::cout << "MSE: " << mse << std::endl;
    std::cout << "RMSE: " << rmse << std::endl;
    std::cout << "NRMSE: " << nrmse << std::endl;
    std::cout << "PSNR (dB): " << psnr << std::endl;
    writeVectorBinary(compressed, fileName + ".comp");
    writeVectorBinary(losslessValues, fileName + ".lossless");
    writeMap(codeTable, fileName + ".codebook");
}

template <typename T, size_t face_dim, size_t node_dim>
void DFSCompressOneSequence(const T xi, const T *nodeCoors, const size_t *cells, const std::unordered_map<size_t, std::unordered_set<size_t>> &node2Cells, std::vector<std::vector<size_t>> &cellNeighbors, const T *values, const size_t numNodes, const size_t seed, bool *visitedCells, RandomSet<size_t> &unvisitedCells, bool *visitedNodes, size_t &numVisitedNodes, std::vector<int> &quantCodes, T *decompValues)
{
    RandomSet<std::pair<size_t, size_t>> stack({{seed, numNodes}});
    while (stack.size() > 0)
    {
        auto [currCellId, prevCellId] = stack.popBack();

        if (!visitedCells[currCellId])
        {

            // determine & predict the newly introduced node
            size_t currNodeId = currCellId * (face_dim + 1); // node to be updated
            size_t prevNodeId = prevCellId * (face_dim + 1);
            for (size_t i = 0; i < face_dim + 1; i++)
            {
                bool commonNode = false;
                for (size_t j = 0; j < face_dim + 1; j++)
                {
                    if (cells[currNodeId + i] == cells[prevNodeId + j])
                    {
                        commonNode = true;
                        break;
                    }
                }
                if (!commonNode)
                {
                    currNodeId = cells[currNodeId + i];
                    break;
                }
            }

            T predValue = barycentricExtrap<T, face_dim, node_dim>(currNodeId, prevCellId, nodeCoors, cells, decompValues);
            T dist = (values[currNodeId] - predValue) / (2 * xi);
            int quantCode = (1 << (m - 1)) + round(dist);
            if (quantCode < 1 || quantCode > (1 << m) - 1)
            {
                // std::cout << "Unpredictable!" << std::endl;
                break; // unpredictable; early stop
            }
            else
            {
                decompValues[currNodeId] = predValue + (quantCode - (1 << (m - 1))) * 2 * xi;
                quantCodes.push_back(quantCode);
                visitedUpdate<face_dim>(currNodeId, cells, node2Cells, visitedCells, unvisitedCells, visitedNodes, numVisitedNodes);
            }
        }
        std::vector<size_t> neighbors = cellNeighbors[currCellId];
        std::sort(neighbors.begin(), neighbors.end(), std::greater<>());
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

template <typename T, size_t face_dim, size_t node_dim>
void DFSDecompression(const T xi, const T *nodeCoors, const size_t *cells, const size_t numCells, const size_t numNodes, const size_t randomSeed, const std::string &fileName)
{
    bool visitedCells[numCells] = {};
    RandomSet<size_t> unvisitedCells(numCells);
    bool visitedNodes[numNodes] = {};
    size_t numVisitedNodes = 0;
    T *decompValues = new T[numNodes];
    std::unordered_map<size_t, std::unordered_set<size_t>> node2Cells;
    buildNode2CellsMap<face_dim>(cells, numCells, node2Cells);
    std::vector<std::vector<size_t>> cellNeighbors;
    buildCellNeighbors<face_dim>(cells, numCells, cellNeighbors);
    std::vector<uint8_t> compressed = readVectorBinary<uint8_t>(fileName + ".comp");
    std::vector<T> losslessValues = readVectorBinary<T>(fileName + ".lossless");
    std::map<int, std::pair<uint32_t, int>> codeTable = readMap(fileName + ".codebook");
    std::vector<int> quantCodes = huffmanZstdDecompress(compressed, codeTable, 234295 * sizeof(int), 234295);
    size_t numSeeds = losslessValues.size() / (face_dim + 1);
    size_t losslessId = 0;
    size_t quantCodeId = 0;
    srand(randomSeed);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < numSeeds; i++)
    {
        // pop a random seed cell from unvisited cells
        size_t seed = unvisitedCells.popRandom();
        visitedCells[seed] = true;

        // set all nodes of the seed as visited
        size_t startNodeId = seed * (face_dim + 1);
        for (size_t j = 0; j < face_dim + 1; j++)
        {
            size_t nodeId = cells[startNodeId + j];
            visitedUpdate<face_dim>(nodeId, cells, node2Cells, visitedCells, unvisitedCells, visitedNodes, numVisitedNodes);
            decompValues[nodeId] = losslessValues[losslessId];
            losslessId++;
        }

        // reconstruct a sequence of cells starting from the seed
        DFSDecompressOneSequence<T, face_dim, node_dim>(xi, nodeCoors, cells, node2Cells, cellNeighbors, numCells, numNodes, seed, visitedCells, unvisitedCells, visitedNodes, numVisitedNodes, quantCodes, quantCodeId, decompValues);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Decompression time: " << duration.count() << " ms" << std::endl;
    writeRawArrayBinary(decompValues, numNodes, fileName + ".ptz.out");
}

template <typename T, size_t face_dim, size_t node_dim>
void DFSDecompressOneSequence(const T xi, const T *nodeCoors, const size_t *cells, const std::unordered_map<size_t, std::unordered_set<size_t>> &node2Cells, std::vector<std::vector<size_t>> &cellNeighbors, const size_t numCells, const size_t numNodes, const size_t seed, bool *visitedCells, RandomSet<size_t> &unvisitedCells, bool *visitedNodes, size_t &numVisitedNodes, const std::vector<int> &quantCodes, size_t &quantCodeId, T *decompValues)
{
    RandomSet<std::pair<size_t, size_t>> stack({{seed, numNodes}});
    int quantCodeMax = 1 << m;
    int quantCode = quantCodes[quantCodeId];
    while (quantCode < quantCodeMax)
    {
        auto [currCellId, prevCellId] = stack.popBack();

        if (!visitedCells[currCellId])
        {
            size_t currNodeId = currCellId * (face_dim + 1); // node to be updated
            size_t prevNodeId = prevCellId * (face_dim + 1);
            for (size_t i = 0; i < face_dim + 1; i++)
            {
                bool commonNode = false;
                for (size_t j = 0; j < face_dim + 1; j++)
                {
                    if (cells[currNodeId + i] == cells[prevNodeId + j])
                    {
                        commonNode = true;
                        break;
                    }
                }
                if (!commonNode)
                {
                    currNodeId = cells[currNodeId + i];
                    break;
                }
            }

            T predValue = barycentricExtrap<T, face_dim, node_dim>(currNodeId, prevCellId, nodeCoors, cells, decompValues);
            decompValues[currNodeId] = predValue + (quantCode - (1 << (m - 1))) * 2 * xi;
            visitedUpdate<face_dim>(currNodeId, cells, node2Cells, visitedCells, unvisitedCells, visitedNodes, numVisitedNodes);
            quantCodeId++;
            quantCode = quantCodes[quantCodeId];
        }
        std::vector<size_t> neighbors = cellNeighbors[currCellId];
        std::sort(neighbors.begin(), neighbors.end(), std::greater<>());
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

int main(int argc, char *argv[])
{
    Parsing(argc, argv);

    size_t *cells = new size_t[numCells * (face_dim + 1)];
    int *tempCells = new int[numCells * (face_dim + 1)];
    readRawArrayBinary(cellsFile, tempCells, numCells * (face_dim + 1), DataType::INT);
    for (size_t i = 0; i < numCells * (face_dim + 1); i++)
    {
        cells[i] = static_cast<size_t>(tempCells[i]);
    }

    if (isDouble)
    {
        double *nodeCoors = new double[numNodes * node_dim];
        double *values = new double[numNodes];
        readRawArrayBinary(nodesCoorFile, nodeCoors, numNodes * node_dim, DataType::DOUBLE);
        readRawArrayBinary(orgDataFile, values, numNodes, DataType::DOUBLE);
        double xi = err_bound;
        std::cout << isABS << " " << xi << std::endl;
        if (!isABS)
        {
            double range = getRange(values, numNodes);
            xi *= range;
        }
        if (face_dim == 2)
        {
            if (node_dim == 2)
            {
                if (!orgDataFile.empty())
                {
                    DFSCompression<double, 2, 2>(xi, nodeCoors, cells, values, numCells, numNodes, randomSeed, compressedFile);
                }
                if (!decompressedFile.empty())
                {
                    DFSDecompression<double, 2, 2>(xi, nodeCoors, cells, numCells, numNodes, randomSeed, decompressedFile);
                }
            }
            else
            {
                if (!orgDataFile.empty())
                {
                    DFSCompression<double, 2, 3>(xi, nodeCoors, cells, values, numCells, numNodes, randomSeed, compressedFile);
                }
                if (!decompressedFile.empty())
                {
                    DFSDecompression<double, 2, 3>(xi, nodeCoors, cells, numCells, numNodes, randomSeed, decompressedFile);
                }
            }
        }
        else
        {
            if (node_dim == 3)
            {
                if (!orgDataFile.empty())
                {
                    DFSCompression<double, 3, 3>(xi, nodeCoors, cells, values, numCells, numNodes, randomSeed, compressedFile);
                }
                if (!decompressedFile.empty())
                {
                    DFSDecompression<double, 3, 3>(xi, nodeCoors, cells, numCells, numNodes, randomSeed, decompressedFile);
                }
            }
            else
            {
                parseError("vertex dim and cell dim not match");
            }
        }
    }
    else
    {

        float *nodeCoors = new float[numNodes * node_dim];
        float *values = new float[numNodes];
        readRawArrayBinary(nodesCoorFile, nodeCoors, numNodes * node_dim, DataType::FLOAT);
        readRawArrayBinary(orgDataFile, values, numNodes, DataType::FLOAT);
        float xi = err_bound;
        if (!isABS)
        {
            float range = getRange(values, numNodes);
            xi *= range;
        }

        if (face_dim == 2)
        {
            if (node_dim == 2)
            {
                if (!orgDataFile.empty())
                {
                    DFSCompression<float, 2, 2>(xi, nodeCoors, cells, values, numCells, numNodes, randomSeed, compressedFile);
                }
                if (!decompressedFile.empty())
                {
                    DFSDecompression<float, 2, 2>(xi, nodeCoors, cells, numCells, numNodes, randomSeed, decompressedFile);
                }
            }
            else
            {
                if (!orgDataFile.empty())
                {
                    DFSCompression<float, 2, 3>(xi, nodeCoors, cells, values, numCells, numNodes, randomSeed, compressedFile);
                }
                if (!decompressedFile.empty())
                {
                    DFSDecompression<float, 2, 3>(xi, nodeCoors, cells, numCells, numNodes, randomSeed, decompressedFile);
                }
            }
        }
        else
        {
            if (node_dim == 3)
            {
                if (!orgDataFile.empty())
                {
                    DFSCompression<float, 3, 3>(xi, nodeCoors, cells, values, numCells, numNodes, randomSeed, compressedFile);
                }
                if (!decompressedFile.empty())
                {
                    DFSDecompression<float, 3, 3>(xi, nodeCoors, cells, numCells, numNodes, randomSeed, decompressedFile);
                }
            }
            else
            {
                parseError("vertex dim and cell dim not match");
            }
        }
    }
}
