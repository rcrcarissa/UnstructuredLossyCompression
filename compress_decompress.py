import numpy as np
import networkx as nx
import pickle
import trimesh
from dahuffman import HuffmanCodec
import zstd
import time
import matplotlib.pyplot as plt
import argparse
from cmse.cmse import *

m = 16


def preprocessing_write(dataset, attribute):
    fname = "datasets/" + dataset + "/" + dataset + ".pickle"
    with open(fname, "rb") as f:
        data = pickle.load(f)
    nodes_coor = data["nodes_coor"]
    cells = data["cells"]
    func = data[attribute]
    num_cells = cells.shape[0]
    dual_graph = dualGraph4MeshTetCells(cells, num_cells)
    dim = nodes_coor.shape[1]
    node2cells = node2cellsDict(cells)
    func.tofile("datasets/" + dataset + "/" + attribute + "_binary.dat")
    with open("datasets/" + dataset + "/" + "dual.pickle", "wb") as f:
        pickle.dump(dual_graph, f)
    with open("datasets/" + dataset + "/" + "node2cells.pickle", "wb") as f:
        pickle.dump(node2cells, f)
    return nodes_coor, cells, func, dual_graph, node2cells, dim


def preprocessing_read(dataset, attribute):
    fname = "datasets/" + dataset + "/" + dataset + ".pickle"
    with open(fname, "rb") as f:
        data = pickle.load(f)
    nodes_coor = data["nodes_coor"]
    cells = data["cells"]
    func = np.fromfile("datasets/" + dataset + "/" + attribute + "_binary.dat")
    dim = nodes_coor.shape[1]
    with open("datasets/" + dataset + "/" + "dual.pickle", "rb") as f:
        dual_graph = pickle.load(f)
    with open("datasets/" + dataset + "/" + "node2cells.pickle", "rb") as f:
        node2cells = pickle.load(f)
    return nodes_coor, cells, func, dual_graph, node2cells, dim


def plotMetrics(dataset, attribute, xi_pct_range, eval_metrics):
    plot_range = 100 * np.array(xi_pct_range)
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    axs[0].plot(plot_range, eval_metrics["runtime"][:, 0], c='dodgerblue', label="compression")
    axs[0].plot(plot_range, eval_metrics["runtime"][:, 1], c='dodgerblue', linestyle="dashed", label="decompression")
    axs[0].set_xlabel("error bound (%)")
    axs[0].set_ylabel("runtime")
    axs[0].legend()
    axs[1].plot(plot_range, eval_metrics["nrmse"], c='dodgerblue', label="NRMSE")
    axs[1].plot(plot_range, eval_metrics["cnrmse"], c='dodgerblue', linestyle="dashed", label="CNRMSE")
    axs[1].set_xlabel("error bound (%)")
    axs[1].set_ylabel("NRMSE / CNRMSE")
    axs[2].plot(eval_metrics["br"], eval_metrics["psnr"], c='dodgerblue', label="PSNR")
    axs[2].plot(eval_metrics["br"], eval_metrics["cpsnr"], c='dodgerblue', linestyle="dashed", label="CPSNR")
    axs[2].set_xlabel("bit rate")
    axs[2].set_ylabel("PSNR / CPSNR (dB)")
    axs[3].plot(plot_range, eval_metrics["cr"])
    axs[3].set_xlabel("error bound (%)")
    axs[3].set_ylabel("compression ratio")
    fig.suptitle(dataset + " data - " + attribute)
    plt.savefig("results/" + dataset + "/" + dataset + "_" + attribute + ".png")


def dualGraph4MeshTetCells(cells, num_cells):
    # create cell2cells dict
    cell2cell_dict = {}
    for i, cl in enumerate(cells):
        c = np.sort(cl)
        try:
            cell2cell_dict[(c[1], c[2], c[3])].append(i)
        except:
            cell2cell_dict[(c[1], c[2], c[3])] = [i]
        try:
            cell2cell_dict[(c[0], c[2], c[3])].append(i)
        except:
            cell2cell_dict[(c[0], c[2], c[3])] = [i]
        try:
            cell2cell_dict[(c[0], c[1], c[3])].append(i)
        except:
            cell2cell_dict[(c[0], c[1], c[3])] = [i]
        try:
            cell2cell_dict[(c[0], c[1], c[2])].append(i)
        except:
            cell2cell_dict[(c[0], c[1], c[2])] = [i]

    # construct dual graph
    dual_graph = nx.Graph()
    dual_graph.add_nodes_from(range(num_cells))
    for cell, cells in cell2cell_dict.items():
        if len(cells) == 2:
            dual_graph.add_edge(cells[0], cells[1])

    return dual_graph


def dualGraph4MeshTriCells(cells, num_cells):
    # create edge2triangle dict
    edge2tri_dict = {}
    for i, cl in enumerate(cells):
        c = np.sort(cl)
        try:
            edge2tri_dict[(c[0], c[1])].append(i)
        except:
            edge2tri_dict[(c[0], c[1])] = [i]
        try:
            edge2tri_dict[(c[0], c[2])].append(i)
        except:
            edge2tri_dict[(c[0], c[2])] = [i]
        try:
            edge2tri_dict[(c[1], c[2])].append(i)
        except:
            edge2tri_dict[(c[1], c[2])] = [i]

    # construct dual graph
    dual_graph = nx.Graph()
    dual_graph.add_nodes_from(range(num_cells))
    for e, tris in edge2tri_dict.items():
        if len(tris) == 2:
            dual_graph.add_edge(tris[0], tris[1])

    return dual_graph


def node2cellsDict(cells):
    node2cells = {}
    for i, c in enumerate(cells):
        for n in c:
            try:
                node2cells[n].append(i)
            except:
                node2cells[n] = [i]
    return node2cells


def barycentric_extrap(point_id, cell_id, nodes_coor, cells, values, dim):
    if dim == 2:
        nodes_of_cell = cells[cell_id]
        cell_nodes_coor = np.zeros((1, 3, 3))
        point_dim = nodes_coor.shape[1]
        cell_nodes_coor[0, 0, :point_dim] = nodes_coor[nodes_of_cell[0]]
        cell_nodes_coor[0, 1, :point_dim] = nodes_coor[nodes_of_cell[1]]
        cell_nodes_coor[0, 2, :point_dim] = nodes_coor[nodes_of_cell[2]]
        point_coor = np.zeros(3)
        point_coor[:point_dim] = nodes_coor[point_id]
        barycentric_coor = trimesh.triangles.points_to_barycentric(cell_nodes_coor, [point_coor])
        return values[nodes_of_cell[0]] * barycentric_coor[0][0] + values[nodes_of_cell[1]] * barycentric_coor[0][1] + \
            values[nodes_of_cell[2]] * barycentric_coor[0][2]
    T = np.zeros((3, 3))
    for i in range(3):
        T[:, i] = nodes_coor[cells[cell_id][i]] - nodes_coor[cells[cell_id][-1]]
    a0, a1, a2 = np.dot(np.linalg.inv(T), nodes_coor[point_id] - nodes_coor[cells[cell_id][-1]])
    a3 = 1 - a0 - a1 - a2
    return a0 * values[cells[cell_id][0]] + a1 * values[cells[cell_id][1]] + a2 * values[cells[cell_id][2]] + a3 * \
        values[cells[cell_id][3]]


def visitedUpdate(newly_visited_node: int, visited_cells: np.array, unvisited_cells: set, visited_nodes: np.array,
                  num_visited_nodes: int, cells: np.array, node2cells: dict):
    if visited_nodes[newly_visited_node] == 0:
        visited_nodes[newly_visited_node] = 1
        num_visited_nodes += 1
    incident_cells = node2cells[newly_visited_node]
    for cell in incident_cells:
        if visited_cells[cell] == 0:
            if_cell_visited = [visited_nodes[n] for n in cells[cell]]
            if all(if_cell_visited):
                visited_cells[cell] = 1
                unvisited_cells.remove(cell)
    return visited_cells, unvisited_cells, visited_nodes, num_visited_nodes


def DFS_compression(graph: nx.Graph, xi: float, dim: int, nodes_coor: np.array, cells: np.array, node2cells: dict,
                    func: np.array):
    '''
    This function conducts DFS and compression on dual graph of a mesh with two conditions:
    1) Early stop: if the newly interpolated mesh node is unpredictable;
    2) Visited constraint: if all mesh nodes of a cell is interpolated, mark the cell as visited.

    Augments:
        graph: dual graph of the mesh
        node2cells: a mapping from a mesh node to all cells incident to it
        func: function values on mesh nodes

    Return:
    '''

    # initialization
    num_cells = len(cells)
    num_nodes = len(nodes_coor)
    visited_cells = np.zeros(num_cells)
    unvisited_cells = set(range(num_cells))
    visited_nodes = np.zeros(num_nodes)
    num_visited_nodes = 0
    decomp_func = np.zeros(num_nodes)
    data2compressed = []
    sequences = {}

    while num_visited_nodes < num_nodes:
        seed = unvisited_cells.pop()
        visited_cells[seed] = 1
        for n in cells[seed]:
            visited_cells, unvisited_cells, visited_nodes, num_visited_nodes = \
                visitedUpdate(n, visited_cells, unvisited_cells, visited_nodes, num_visited_nodes, cells, node2cells)
            data2compressed.append(func[n])
            decomp_func[n] = func[n]
        data2compressed, sequence, visited_cells, unvisited_cells, visited_nodes, num_visited_nodes, decomp_func = \
            DFS_compression_oneSeed(data2compressed, xi, dim, graph, nodes_coor, cells, node2cells, seed, visited_cells,
                                    unvisited_cells, visited_nodes, num_visited_nodes, func, decomp_func)
        sequences[seed] = sequence

    # for every seed: four values at seed (float64) and end mark of sequences (float32)
    unpredicted_data_size = len(sequences) * (4 * 8 + 4)
    # Huffman encoding
    codec = HuffmanCodec.from_data(data2compressed)
    encoded = codec.encode(data2compressed)
    compressed = zstd.compress(encoded, 22)
    compressed_size = unpredicted_data_size + len(compressed)
    compression_ratio = (num_nodes - 4 * len(sequences)) * 8 / compressed_size

    return sequences, compressed, compression_ratio, codec


def DFS_compression_oneSeed(data2compressed: list, xi: float, dim: int, graph: nx.Graph, nodes_coor: np.array,
                            cells: np.array, node2cells: dict, seed: int, visited_cells: np.array,
                            unvisited_cells: set, visited_nodes: np.array, num_visited_nodes: int, func: np.array,
                            decomp_func: np.array):
    '''
    This function conducts DFS and compression for ONE SEED on dual graph of a cellangular mesh with two conditions:
    1) Early stop: if the newly interpolated mesh node is unpredictable;
    2) Visited constraint: if all three mesh nodes of a cell is interpolated, mark the cell as visited.

    Augments:
        graph: dual graph of the mesh
        mesh: original mesh
        node2cells: a mapping from a mesh node to all cells incident to it
        seed: index of a seed cell
        visited_cells: if cells are visited
        visited_nodes: if nodes are visited
        func: function values on mesh nodes
        decomp_func: decompressed function values on mesh nodes

    Return:
        sequence: order of visited cells by a seed
        quantization_code: a sequence of quantization codes in list format
    '''
    stack = [(seed, None)]
    stack_set = {(seed, None)}
    sequence = []
    while len(stack) > 0:
        current_cell, prev_cell = stack.pop()
        stack_set.remove((current_cell, prev_cell))
        if visited_cells[current_cell] == 0:  # visited constraint
            node_to_update = list(set(cells[current_cell]) - set(cells[prev_cell]))[0]
            pred_f = barycentric_extrap(node_to_update, prev_cell, nodes_coor, cells, decomp_func, dim)  # predictor
            code_distance = int((func[node_to_update] - pred_f) / xi)  # quantization
            if code_distance < 0:
                code = 2 ** (m - 1) + int(np.floor(code_distance / 2))
            else:
                code = 2 ** (m - 1) + int(np.ceil(code_distance / 2))
            if code < 1 or code > 2 ** m - 1:
                break  # unpredictable; early stop
            else:
                decomp_func[node_to_update] = pred_f + (code - 2 ** (m - 1)) * 2 * xi
                sequence.append(current_cell)
                data2compressed.append(code)
                visited_cells, unvisited_cells, visited_nodes, num_visited_nodes = \
                    visitedUpdate(node_to_update, visited_cells, unvisited_cells, visited_nodes, num_visited_nodes,
                                  cells, node2cells)
        neighbors = list(graph.neighbors(current_cell))
        neighbors.sort(reverse=True)  # search strategy: start with cell with min index
        for t in neighbors:
            if visited_cells[t] == 0 and (t, current_cell) not in stack_set:
                stack.append((t, current_cell))
                stack_set.add((t, current_cell))
    data2compressed.append(np.inf)
    return data2compressed, sequence, visited_cells, unvisited_cells, visited_nodes, num_visited_nodes, decomp_func


def DFS_decompression(compressed: list, xi: float, dim: int, codec: HuffmanCodec, graph: nx.Graph, nodes_coor: np.array,
                      cells: np.array, node2cells: dict):
    '''
    This function conducts DFS and decompression on dual graph of a cellangular mesh with two conditions:
    1) Early stop: if the newly interpolated mesh node is unpredictable;
    2) Visited constraint: if all three mesh nodes of a cell is interpolated, mark the cell as visited.

    Augments:
        compressed: a sequence of compressed code
        sequences: a dictionary mapping seeds to cells visited by it
        graph: dual graph of the mesh
        mesh: original mesh
        node2cells: a mapping from a mesh node to all cells incident to it

    Return:

    '''

    # initialization
    num_cells = len(cells)
    num_nodes = len(nodes_coor)
    visited_cells = np.zeros(num_cells)
    unvisited_cells = set(range(num_cells))
    visited_nodes = np.zeros(num_nodes)
    decomp_func = np.zeros(num_nodes)
    encoded = zstd.decompress(compressed)
    decompressed_data = np.array(codec.decode(encoded))
    num_visited_nodes = 0
    seed_values = []
    decompressed_sequence = []
    streamline_count = 0

    for i, v in enumerate(decompressed_data):
        if streamline_count == 0:
            prev_v = decompressed_data[i - 1]
            if prev_v == np.inf:
                seed_values.append(decompressed_data[i:i + dim + 1])
                if decompressed_data[i + dim + 1] == np.inf:
                    streamline_count = dim + 1
                    decompressed_sequence.append([])
                else:
                    streamline_count = dim
            elif v == np.inf:
                pass
            else:
                if decompressed_data[i - dim - 2] == np.inf:
                    decompressed_sequence.append([np.int64(v)])
                else:
                    decompressed_sequence[-1].append(np.int64(v))
        else:
            streamline_count -= 1
    num_seeds = len(seed_values)

    for i in range(num_seeds):
        seed = unvisited_cells.pop()
        visited_cells[seed] = 1
        for j, n in enumerate(cells[seed]):
            visited_cells, unvisited_cells, visited_nodes, num_visited_nodes = \
                visitedUpdate(n, visited_cells, unvisited_cells, visited_nodes, num_visited_nodes, cells, node2cells)
            decomp_func[n] = seed_values[i][j]
        visited_cells, unvisited_cells, visited_nodes, num_visited_nodes, decomp_func = \
            DFS_decompression_oneSeed(decompressed_sequence[i], xi, dim, graph, nodes_coor, cells, node2cells, seed,
                                      visited_cells, unvisited_cells, visited_nodes, num_visited_nodes, decomp_func)
    return decomp_func


def DFS_decompression_oneSeed(decompressed_sequence: list, xi: float, dim: int, graph: nx.Graph,
                              nodes_coor: np.array, cells: np.array, node2cells: dict, seed: int,
                              visited_cells: np.array, unvisited_cells: set, visited_nodes: np.array,
                              num_visited_nodes: int, decomp_func: np.array):
    '''
    This function conducts DFS and decompression for ONE SEED on dual graph of a cellangular mesh with two conditions:
    1) Early stop: if the newly interpolated mesh node is unpredictable;
    2) Visited constraint: if all three mesh nodes of a cell is interpolated, mark the cell as visited.

    Augments:
        compressed: a list of quantization codes following the seed
        graph: dual graph of the mesh
        mesh: original mesh
        node2cells: a mapping from a mesh node to all cells incident to it
        seed: index of a seed cell
        visited_cells: if cells are visited
        visited_nodes: if nodes are visited
        decomp_func: decomposed function values on mesh nodes

    Return:

    '''
    stack = [(seed, None)]
    stack_set = {(seed, None)}
    sequence_length = len(decompressed_sequence)
    current_length = 0
    while current_length < sequence_length:
        current_cell, prev_cell = stack.pop()
        stack_set.remove((current_cell, prev_cell))
        if visited_cells[current_cell] == 0:  # visited constraint
            node_to_update = list(set(cells[current_cell]) - set(cells[prev_cell]))[0]
            pred_f = barycentric_extrap(node_to_update, prev_cell, nodes_coor, cells, decomp_func, dim)
            code = decompressed_sequence[current_length]
            code_distance = code - 2 ** (m - 1)
            visited_cells, unvisited_cells, visited_nodes, num_visited_nodes = \
                visitedUpdate(node_to_update, visited_cells, unvisited_cells, visited_nodes, num_visited_nodes, cells,
                              node2cells)
            decomp_func[node_to_update] = pred_f + code_distance * 2 * xi
            current_length += 1
        neighbors = list(graph.neighbors(current_cell))
        neighbors.sort(reverse=True)  # search strategy: start with cell with min index
        for t in neighbors:
            if visited_cells[t] == 0 and (t, current_cell) not in stack_set:
                stack.append((t, current_cell))
                stack_set.add((t, current_cell))
    return visited_cells, unvisited_cells, visited_nodes, num_visited_nodes, decomp_func


if __name__ == "__main__":
    xi_pct_range = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6]

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str)
    parser.add_argument('-attribute', type=str)
    args = parser.parse_args()
    dataset = args.dataset
    attribute = args.attribute

    t0 = time.perf_counter()
    nodes_coor, cells, func, dual, node2cells, dim = preprocessing_write(dataset, attribute)
    # nodes_coor, cells, func, dual, node2cells, dim = preprocessing_read(dataset, attribute)
    num_nodes = nodes_coor.shape[0]
    max_signal = np.max(func)
    min_signal = np.min(func)
    xi_dim = len(xi_pct_range)
    runtime = np.zeros((xi_dim, 2))
    mse = np.zeros(xi_dim)
    rmse = np.zeros(xi_dim)
    nrmse = np.zeros(xi_dim)
    psnr = np.zeros(xi_dim)
    cmse = np.zeros(xi_dim)
    crmse = np.zeros(xi_dim)
    cnrmse = np.zeros(xi_dim)
    cpsnr = np.zeros(xi_dim)
    compression_ratio = np.zeros(xi_dim)
    bit_rate = np.zeros(xi_dim)
    t1 = time.perf_counter()
    print("Preprocessing Done! Time:", t1 - t0)

    for i, xi_pct in enumerate(xi_pct_range):
        xi = xi_pct * (max_signal - min_signal)
        t0 = time.perf_counter()
        sequences, compressed_codes, cr, codec = DFS_compression(dual, xi, dim, nodes_coor, cells, node2cells, func)
        t1 = time.perf_counter()
        decompressed_vals = DFS_decompression(compressed_codes, xi, dim, codec, dual, nodes_coor, cells, node2cells)
        t2 = time.perf_counter()
        runtime[i, 0] = t1 - t0
        runtime[i, 1] = t2 - t1
        compression_ratio[i] = cr
        bit_rate[i] = 64 / cr
        mse[i] = np.sum((decompressed_vals - func) ** 2) / num_nodes
        rmse[i] = np.sqrt(mse[i])
        nrmse[i] = rmse[i] / (max_signal - min_signal)
        psnr[i] = 20 * np.log10(max_signal - min_signal) - 10 * np.log10(mse[i])
        if dim == 2:
            cmse[i] = triangularMeshCMSE(nodes_coor, cells, func, decompressed_vals)
        else:
            cmse[i] = tetraheronMeshCMSE(nodes_coor, cells, func, decompressed_vals)
        crmse[i] = np.sqrt(cmse[i])
        cnrmse[i] = crmse[i] / (max_signal - min_signal)
        cpsnr[i] = - 20 * np.log10(cnrmse[i])

        with open("results/" + dataset + "/" + attribute + "_compressed_" + str(xi_pct) + ".dat", "wb") as f:
            f.write(compressed_codes)
        with open("results/" + dataset + "/" + attribute + "_sequences_" + str(xi_pct) + ".pickle", "wb") as f:
            pickle.dump(sequences, f)
        with open("results/" + dataset + "/" + attribute + "_decompressed_vals_" + str(xi_pct) + ".pickle", "wb") as f:
            pickle.dump(decompressed_vals, f)

        print(t1 - t0, t2 - t1)
        print(i, xi_pct)

    eval_metrics = {"runtime": runtime, "mse": mse, "rmse": rmse, "nrmse": nrmse, "psnr": psnr, "cmse": cmse,
                    "crmse": crmse, "cnrmse": cnrmse, "cpsnr": cpsnr, "cr": compression_ratio, "br": bit_rate}
    plotMetrics(dataset, attribute, xi_pct_range, eval_metrics)
    with open("results/" + dataset + "/" + attribute + "_eval_metrics.pickle", "wb") as f:
        pickle.dump(eval_metrics, f)
