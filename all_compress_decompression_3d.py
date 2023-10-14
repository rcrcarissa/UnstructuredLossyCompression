import numpy as np
import networkx as nx
import trimesh
import vtk
import pickle
from dahuffman import HuffmanCodec
import zstd
import time
import matplotlib.pyplot as plt

seed_instance = 87
m = 16
xi_range = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6]


def dualGraph4MeshTetCells(cells, num_cells):
    # create tet2tetrahedra dict
    tet2tet_dict = {}
    for i, cl in enumerate(cells):
        c = np.sort(cl)
        try:
            tet2tet_dict[(c[1], c[2], c[3])].append(i)
        except:
            tet2tet_dict[(c[1], c[2], c[3])] = [i]
        try:
            tet2tet_dict[(c[0], c[2], c[3])].append(i)
        except:
            tet2tet_dict[(c[0], c[2], c[3])] = [i]
        try:
            tet2tet_dict[(c[0], c[1], c[3])].append(i)
        except:
            tet2tet_dict[(c[0], c[1], c[3])] = [i]
        try:
            tet2tet_dict[(c[0], c[1], c[2])].append(i)
        except:
            tet2tet_dict[(c[0], c[1], c[2])] = [i]

    # construct dual graph
    dual_graph = nx.Graph()
    dual_graph.add_nodes_from(range(num_cells))
    for tet, tets in tet2tet_dict.items():
        if len(tets) == 2:
            dual_graph.add_edge(tets[0], tets[1])

    return dual_graph


def node2tetsDict(cells, num_nodes):
    node2tets = {}
    for n in range(num_nodes):
        node2tets[n] = []
    for i, c in enumerate(cells):
        for n in c:
            node2tets[n].append(i)
    return node2tets


def tet_barycentetc_interp(point_id, tet_id, nodes_coor, cells, values):
    T = np.zeros((3, 3))
    for i in range(3):
        T[:, i] = nodes_coor[cells[tet_id][i]] - nodes_coor[cells[tet_id][3]]
    a0, a1, a2 = np.dot(np.linalg.inv(T), nodes_coor[point_id] - nodes_coor[cells[tet_id][3]])
    a3 = 1 - a0 - a1 - a2
    return a0 * values[cells[tet_id][0]] + a1 * values[cells[tet_id][1]] + a2 * values[cells[tet_id][2]] + a3 * values[
        cells[tet_id][3]]


def visitedUpdate(newly_visited_node: int, visited_tets: np.array, visited_nodes: np.array,
                  cells: np.array, node2tets: dict):
    visited_nodes[newly_visited_node] = 1
    tets = node2tets[newly_visited_node]
    for tet in tets:
        if_tet_visited = [visited_nodes[n] for n in cells[tet]]
        if all(v == 1 for v in if_tet_visited):
            visited_tets[tet] = 1
    return visited_tets, visited_nodes


def DFS_compression(graph: nx.Graph, xi: float, mesh: trimesh.Trimesh, node2tets: dict, func: np.array):
    '''
    This function conducts DFS and compression on dual graph of a tetangular mesh with two conditions:
    1) Early stop: if the newly interpolated mesh node is unpredictable;
    2) Visited constraint: if all three mesh nodes of a tetrahedron is interpolated, mark the tetrahedron as visited.

    Augments:
        graph: dual graph of the mesh
        mesh: original mesh
        node2tets: a mapping from a mesh node to all tetrahedra incident to it
        func: function values on mesh nodes

    Return:
    '''

    # initialization
    num_cells = len(mesh.faces)
    num_nodes = len(mesh.vertices)
    visited_tetrahedra = np.zeros(num_cells)
    visited_nodes = np.zeros(num_nodes)
    decomp_func = np.zeros(num_nodes)
    np.random.seed(seed_instance)
    sequences = {}
    quantization_codes = []

    while not all(visited_nodes):
        unvisited_tetrahedra = np.where(visited_tetrahedra == 0)[0]
        seed = np.random.choice(unvisited_tetrahedra, 1)[0]
        for n in mesh.faces[seed]:
            visited_tetrahedra, visited_nodes = visitedUpdate(n, visited_tetrahedra, visited_nodes, mesh, node2tets)
            decomp_func[n] = func[n]
        sequence, quantization_code, visited_tetrahedra, visited_nodes, decomp_func = \
            DFS_compression_oneSeed(xi, graph, mesh, node2tets, seed, visited_tetrahedra, visited_nodes, func,
                                    decomp_func)
        sequences[seed] = sequence
        quantization_codes += quantization_code

    # two numbers for every seed: values at seed (float) and length of sequence following the seed (int)
    # one number for number of seeds
    # one random seed for seeds generation (int)
    unpredicted_data_size = len(sequences) * (3 * 8 + 4) + 4 + 4
    # Huffman encoding
    data2compressed = [seed_instance, len(sequences)]
    for seed in sequences.keys():
        data2compressed += [func[n] for n in mesh.faces[seed]]
    data2compressed += [len(s) for s in sequences.values()] + quantization_codes
    codec = HuffmanCodec.from_data(data2compressed)
    encoded = codec.encode(data2compressed)
    # codec = HuffmanCodec.from_data(quantization_codes)
    # encoded = codec.encode(quantization_codes)
    compressed = zstd.compress(encoded, 22)
    with open("les_compressed_" + str(xi) + ".les", "wb") as f:
        f.write(compressed)
    compressed_size = unpredicted_data_size + len(compressed)
    compression_ratio = (num_nodes - len(sequences)) * 8 / compressed_size

    return sequences, compressed, compression_ratio, codec


def DFS_compression_oneSeed(xi: float, graph: nx.Graph, mesh: trimesh.Trimesh, node2tets: dict, seed: int,
                            visited_tetrahedra: np.array, visited_nodes: np.array, func: np.array,
                            decomp_func: np.array):
    '''
    This function conducts DFS and compression for ONE SEED on dual graph of a tetangular mesh with two conditions:
    1) Early stop: if the newly interpolated mesh node is unpredictable;
    2) Visited constraint: if all three mesh nodes of a tetrahedron is interpolated, mark the tetrahedron as visited.

    Augments:
        graph: dual graph of the mesh
        mesh: original mesh
        node2tets: a mapping from a mesh node to all tetrahedra incident to it
        seed: index of a seed tetrahedron
        visited_tetrahedra: if tetrahedra are visited
        visited_nodes: if nodes are visited
        func: function values on mesh nodes
        decomp_func: decompressed function values on mesh nodes

    Return:
        sequence: order of visited tetrahedra by a seed
        quantization_code: a sequence of quantization codes in list format
    '''
    stack = [(seed, None)]
    stack_set = {(seed, None)}
    sequence = []
    quantization_code = []
    while len(stack) > 0:
        current_tet, prev_tet = stack.pop()
        stack_set.remove((current_tet, prev_tet))
        if visited_tetrahedra[current_tet] == 0:  # visited constraint
            node_to_update = list(set(mesh.faces[current_tet]) - set(mesh.faces[prev_tet]))[0]
            pred_f = tet_barycentetc_interp(node_to_update, prev_tet, nodes_coor, cells, decomp_func)  # predictor
            code_distance = int((func[node_to_update] - pred_f) / xi)  # quantization
            if code_distance < 0:
                code = 2 ** (m - 1) + int(np.floor(code_distance / 2))
            else:
                code = 2 ** (m - 1) + int(np.ceil(code_distance / 2))
            if code < 1 or code > 2 ** m - 1:
                break  # unpredictable; early stop
            else:
                decomp_func[node_to_update] = pred_f + (code - 2 ** (m - 1)) * 2 * xi
                sequence.append(current_tet)
                quantization_code.append(code)
                visited_tetrahedra, visited_nodes = visitedUpdate(node_to_update, visited_tetrahedra, visited_nodes,
                                                                  mesh, node2tets)
        neighbors = list(graph.neighbors(current_tet))
        neighbors.sort(reverse=True)  # search strategy: start with cell with min index
        for t in neighbors:
            if visited_tetrahedra[t] == 0 and t not in stack_set:
                stack.append((t, current_tet))
                stack_set.add((t, current_tet))
    return sequence, quantization_code, visited_tetrahedra, visited_nodes, decomp_func


def DFS_decompression(compressed: list, xi: float, codec: HuffmanCodec, graph: nx.Graph, mesh: trimesh.Trimesh,
                      node2tets: dict):
    '''
    This function conducts DFS and decompression on dual graph of a tetangular mesh with two conditions:
    1) Early stop: if the newly interpolated mesh node is unpredictable;
    2) Visited constraint: if all three mesh nodes of a tetrahedron is interpolated, mark the tetrahedron as visited.

    Augments:
        compressed: a sequence of compressed code
        sequences: a dictionary mapping seeds to cells visited by it
        graph: dual graph of the mesh
        mesh: original mesh
        node2tets: a mapping from a mesh node to all tetrahedra incident to it

    Return:

    '''

    # initialization
    num_cells = len(mesh.faces)
    num_nodes = len(mesh.vertices)
    visited_tetrahedra = np.zeros(num_cells)
    visited_nodes = np.zeros(num_nodes)
    decomp_func = np.zeros(num_nodes)
    np.random.seed(seed_instance)
    compressed_code_start = 0
    encoded = zstd.decompress(compressed)
    data2compressed = np.array(codec.decode(encoded))
    num_seeds = int(data2compressed[1])
    sequence_lengths = data2compressed[3 * num_seeds + 2:4 * num_seeds + 2]
    compressed_code = data2compressed[4 * num_seeds + 2:]
    seed_idx = 0

    while not all(visited_nodes):
        unvisited_tetrahedra = np.where(visited_tetrahedra == 0)[0]
        seed = np.random.choice(unvisited_tetrahedra, 1)[0]
        for n in mesh.faces[seed]:
            visited_tetrahedra, visited_nodes = visitedUpdate(n, visited_tetrahedra, visited_nodes, mesh, node2tets)
            decomp_func[n] = func[n]
        compressed_code_end = compressed_code_start + int(sequence_lengths[seed_idx])
        visited_tetrahedra, visited_nodes, decomp_func = \
            DFS_decompression_oneSeed(compressed_code[compressed_code_start:compressed_code_end], xi, graph, mesh,
                                      node2tets, seed, visited_tetrahedra, visited_nodes, decomp_func)
        seed_idx += 1
        compressed_code_start = compressed_code_end

    return decomp_func


def DFS_decompression_oneSeed(compressed: list, xi: float, graph: nx.Graph, mesh: trimesh.Trimesh, node2tets: dict,
                              seed: int, visited_tetrahedra: np.array, visited_nodes: np.array, decomp_func: np.array):
    '''
    This function conducts DFS and decompression for ONE SEED on dual graph of a tetangular mesh with two conditions:
    1) Early stop: if the newly interpolated mesh node is unpredictable;
    2) Visited constraint: if all three mesh nodes of a tetrahedron is interpolated, mark the tetrahedron as visited.

    Augments:
        compressed: a list of quantization codes following the seed
        graph: dual graph of the mesh
        mesh: original mesh
        node2tets: a mapping from a mesh node to all tetrahedra incident to it
        seed: index of a seed tetrahedron
        visited_tetrahedra: if tetrahedra are visited
        visited_nodes: if nodes are visited
        decomp_func: decomposed function values on mesh nodes

    Return:

    '''
    stack = [(seed, None)]
    stack_set = {(seed, None)}
    while len(compressed) > 0:
        current_tet, prev_tet = stack.pop()
        stack_set.remove((current_tet, prev_tet))
        if visited_tetrahedra[current_tet] == 0:  # visited constraint
            node_to_update = list(set(mesh.faces[current_tet]) - set(mesh.faces[prev_tet]))[0]
            pred_f = tet_barycentetc_interp(node_to_update, prev_tet, nodes_coor, cells, decomp_func)
            code = compressed[0]
            compressed = np.delete(compressed, 0)
            code_distance = code - 2 ** (m - 1)
            visited_tetrahedra, visited_nodes = visitedUpdate(node_to_update, visited_tetrahedra, visited_nodes, mesh,
                                                              node2tets)
            decomp_func[node_to_update] = pred_f + code_distance * 2 * xi
        neighbors = list(graph.neighbors(current_tet))
        neighbors.sort(reverse=True)  # search strategy: start with cell with min index
        for t in neighbors:
            if visited_tetrahedra[t] == 0 and t not in stack_set:
                stack.append((t, current_tet))
                stack_set.add((t, current_tet))

    return visited_tetrahedra, visited_nodes, decomp_func


if __name__ == "__main__":
    vtk_fname = "res_1d5.org_1_20000.vtk"
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_fname)
    reader.Update()
    data = reader.GetOutput()
    # array = np.array(data.GetPointData().GetAbstractArray("pressure"))
    nodes_coor = np.array(data.GetPoints().GetData())
    num_nodes = nodes_coor.shape[0]
    with open('tetraCells.npy', 'rb') as f:
        cells = np.load(f)
    num_cells = cells.shape[0]
    with open("pressure_1d5.org_1_20000.npy", "rb") as f:
        func = np.load(f)
    dual_graph = dualGraph4MeshTetCells(cells, num_cells)
    mesh = trimesh.Trimesh(nodes_coor, faces=cells)
    node2tets = node2tetsDict(cells, num_nodes)

    max_signal = np.max(func)
    min_signal = np.min(func)
    xi_dim = len(xi_range)
    runtime = np.zeros((xi_dim, 2))
    nrmse = np.zeros(xi_dim)
    psnr = np.zeros(xi_dim)
    compression_ratio = np.zeros(xi_dim)
    bit_rate = np.zeros(xi_dim)

    for i, xi in enumerate(xi_range):
        t0 = time.perf_counter()
        sequences, compressed_codes, cr, codec = DFS_compression(dual_graph, xi, mesh, node2tets, func)
        t1 = time.perf_counter()
        runtime[i, 0] = t1 - t0
        t0 = time.perf_counter()
        decompressed_vals = DFS_decompression(compressed_codes, xi, codec, dual_graph, mesh, node2tets)
        t1 = time.perf_counter()
        runtime[i, 1] = t1 - t0
        compression_ratio[i] = cr
        bit_rate[i] = 64 / cr
        mse = np.sum((decompressed_vals - func) ** 2) / num_nodes
        nrmse[i] = np.sqrt(mse) / (max_signal - min_signal)
        psnr[i] = 20 * np.log10(max_signal) - 10 * np.log10(mse)

        with open("sequences_" + str(xi) + ".pickle", "wb") as f:
            pickle.dump(sequences, f)
        with open("compressed_codes_" + str(xi) + ".pickle", "wb") as f:
            pickle.dump(compressed_codes, f)
        with open("decompressed_vals_" + str(xi) + ".pickle", "wb") as f:
            pickle.dump(decompressed_vals, f)

        print(i)

    fig, axs = plt.subplots(2, 4, figsize=(15, 8))
    axs[0, 0].plot(xi_range, runtime[:, 0])
    axs[0, 0].set_xlabel("global error")
    axs[0, 0].set_ylabel("runtime (compression)")
    axs[0, 1].plot(xi_range, runtime[:, 1])
    axs[0, 1].set_xlabel("global error")
    axs[0, 1].set_ylabel("runtime (decompression)")
    axs[0, 2].plot(xi_range, nrmse)
    axs[0, 2].set_xlabel("global error")
    axs[0, 2].set_ylabel("NRMSE")
    axs[0, 3].plot(xi_range, psnr)
    axs[0, 3].set_xlabel("global error")
    axs[0, 3].set_ylabel("PSNR (dB)")
    axs[1, 0].plot(xi_range, compression_ratio)
    axs[1, 0].set_xlabel("global error")
    axs[1, 0].set_ylabel("compression ratio")
    axs[1, 1].plot(xi_range, bit_rate)
    axs[1, 1].set_xlabel("global error")
    axs[1, 1].set_ylabel("bit rate")
    axs[1, 2].plot(bit_rate, nrmse)
    axs[1, 2].set_xlabel("bit rate")
    axs[1, 2].set_ylabel("NRMSE")
    axs[1, 3].plot(bit_rate, psnr)
    axs[1, 3].set_xlabel("bit rate")
    axs[1, 3].set_ylabel("PSNR (dB)")
    fig.suptitle("LES data")
    plt.savefig("les.png")

    eval_metetcs = {"runtime": runtime, "nrmse": nrmse, "psnr": psnr, "cr": compression_ratio, "br": bit_rate}
    with open("eval_metetcs.pickle", "wb") as f:
        pickle.dump(eval_metetcs, f)
