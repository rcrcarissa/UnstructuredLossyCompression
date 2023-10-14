import netCDF4 as nc
import numpy as np
import networkx as nx
import trimesh
import pickle
from dahuffman import HuffmanCodec
import zstd
import time
import matplotlib.pyplot as plt

seed_instance = 87
m = 16
xi_range = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6]


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


def node2trisDict(cells, num_nodes):
    node2tris = {}
    for n in range(num_nodes):
        node2tris[n] = []
    for i, c in enumerate(cells):
        for n in c:
            node2tris[n].append(i)
    return node2tris


def triangleInterpolation(point_coor, cell_idx, mesh, f):
    nodes_of_cell = mesh.faces[cell_idx]
    cell_nodes_coor = np.zeros((1, 3, 3))
    cell_nodes_coor[0][0] = mesh.vertices[nodes_of_cell[0]]
    cell_nodes_coor[0][1] = mesh.vertices[nodes_of_cell[1]]
    cell_nodes_coor[0][2] = mesh.vertices[nodes_of_cell[2]]
    barycentric_coor = trimesh.triangles.points_to_barycentric(cell_nodes_coor, [[point_coor[0], point_coor[1], 0]])
    return f[nodes_of_cell[0]] * barycentric_coor[0][0] + f[nodes_of_cell[1]] * barycentric_coor[0][1] + f[
        nodes_of_cell[2]] * barycentric_coor[0][2]


def visitedUpdate(newly_visited_node: int, visited_triangles: np.array, visited_nodes: np.array,
                  mesh: trimesh.Trimesh, node2tris: dict):
    visited_nodes[newly_visited_node] = 1
    tris = node2tris[newly_visited_node]
    for tri in tris:
        if_tri_visited = [visited_nodes[n] for n in mesh.faces[tri]]
        if all(v == 1 for v in if_tri_visited):
            visited_triangles[tri] = 1
    return visited_triangles, visited_nodes


def drawVisited(visited_triangles: np.array, visited_nodes: np.array, mesh: trimesh.Trimesh, i: int):
    triangles = np.where(visited_triangles == 1)[0]
    nodes = np.where(visited_nodes == 1)[0]

    mesh_plot = Triangulation(nodes_coor[:, 0], nodes_coor[:, 1])
    fig, ax = plt.subplots(figsize=(20, 20))
    plt.cla()
    ax.triplot(mesh_plot, color='black', linewidth=0.5)

    edges = set()
    for tri in triangles:
        edges.add((mesh.faces[tri][0], mesh.faces[tri][1]))
        edges.add((mesh.faces[tri][0], mesh.faces[tri][2]))
        edges.add((mesh.faces[tri][1], mesh.faces[tri][2]))

    for e in edges:
        ax.plot([mesh.vertices[e[0]][0], mesh.vertices[e[1]][0]], [mesh.vertices[e[0]][1], mesh.vertices[e[1]][1]],
                color="b", linewidth=2)
    for n in nodes:
        ax.plot(mesh.vertices[n][0], mesh.vertices[n][1], 'ro', markersize=5)
    plt.savefig("search_track_fig/" + str(i) + ".png")


def DFS_compression(graph: nx.Graph, xi: float, mesh: trimesh.Trimesh, node2tris: dict, func: np.array):
    '''
    This function conducts DFS and compression on dual graph of a triangular mesh with two conditions:
    1) Early stop: if the newly interpolated mesh node is unpredictable;
    2) Visited constraint: if all three mesh nodes of a triangle is interpolated, mark the triangle as visited.

    Augments:
        graph: dual graph of the mesh
        mesh: original mesh
        node2tris: a mapping from a mesh node to all triangles incident to it
        func: function values on mesh nodes

    Return:
    '''

    # initialization
    num_cells = len(mesh.faces)
    num_nodes = len(mesh.vertices)
    visited_triangles = np.zeros(num_cells)
    visited_nodes = np.zeros(num_nodes)
    decomp_func = np.zeros(num_nodes)
    np.random.seed(seed_instance)
    sequences = {}
    quantization_codes = []

    while not all(visited_nodes):
        unvisited_triangles = np.where(visited_triangles == 0)[0]
        seed = np.random.choice(unvisited_triangles, 1)[0]
        for n in mesh.faces[seed]:
            visited_triangles, visited_nodes = visitedUpdate(n, visited_triangles, visited_nodes, mesh, node2tris)
            decomp_func[n] = func[n]
        sequence, quantization_code, visited_triangles, visited_nodes, decomp_func = \
            DFS_compression_oneSeed(xi, graph, mesh, node2tris, seed, visited_triangles, visited_nodes, func,
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
    with open("MPAS_compressed_" + str(xi) + ".mpas", "wb") as f:
        f.write(compressed)
    compressed_size = unpredicted_data_size + len(compressed)
    compression_ratio = (num_nodes - len(sequences)) * 8 / compressed_size

    return sequences, compressed, compression_ratio, codec


def DFS_compression_oneSeed(xi: float, graph: nx.Graph, mesh: trimesh.Trimesh, node2tris: dict, seed: int,
                            visited_triangles: np.array, visited_nodes: np.array, func: np.array,
                            decomp_func: np.array):
    '''
    This function conducts DFS and compression for ONE SEED on dual graph of a triangular mesh with two conditions:
    1) Early stop: if the newly interpolated mesh node is unpredictable;
    2) Visited constraint: if all three mesh nodes of a triangle is interpolated, mark the triangle as visited.

    Augments:
        graph: dual graph of the mesh
        mesh: original mesh
        node2tris: a mapping from a mesh node to all triangles incident to it
        seed: index of a seed triangle
        visited_triangles: if triangles are visited
        visited_nodes: if nodes are visited
        func: function values on mesh nodes
        decomp_func: decompressed function values on mesh nodes

    Return:
        sequence: order of visited triangles by a seed
        quantization_code: a sequence of quantization codes in list format
    '''
    stack = [(seed, None)]
    stack_set = {(seed, None)}
    sequence = []
    quantization_code = []
    while len(stack) > 0:
        current_tri, prev_tri = stack.pop()
        stack_set.remove((current_tri, prev_tri))
        if visited_triangles[current_tri] == 0:  # visited constraint
            node_to_update = list(set(mesh.faces[current_tri]) - set(mesh.faces[prev_tri]))[0]
            pred_f = triangleInterpolation(mesh.vertices[node_to_update], prev_tri, mesh, decomp_func)  # predictor
            code_distance = int((func[node_to_update] - pred_f) / xi)  # quantization
            if code_distance < 0:
                code = 2 ** (m - 1) + int(np.floor(code_distance / 2))
            else:
                code = 2 ** (m - 1) + int(np.ceil(code_distance / 2))
            if code < 1 or code > 2 ** m - 1:
                break  # unpredictable; early stop
            else:
                decomp_func[node_to_update] = pred_f + (code - 2 ** (m - 1)) * 2 * xi
                sequence.append(current_tri)
                quantization_code.append(code)
                visited_triangles, visited_nodes = visitedUpdate(node_to_update, visited_triangles, visited_nodes, mesh,
                                                                 node2tris)
        neighbors = list(graph.neighbors(current_tri))
        neighbors.sort(reverse=True)  # search strategy: start with cell with min index
        for t in neighbors:
            if visited_triangles[t] == 0 and t not in stack_set:
                stack.append((t, current_tri))
                stack_set.add((t, current_tri))
    return sequence, quantization_code, visited_triangles, visited_nodes, decomp_func


def DFS_decompression(compressed: list, xi: float, codec: HuffmanCodec, graph: nx.Graph, mesh: trimesh.Trimesh,
                      node2tris: dict):
    '''
    This function conducts DFS and decompression on dual graph of a triangular mesh with two conditions:
    1) Early stop: if the newly interpolated mesh node is unpredictable;
    2) Visited constraint: if all three mesh nodes of a triangle is interpolated, mark the triangle as visited.

    Augments:
        compressed: a sequence of compressed code
        sequences: a dictionary mapping seeds to cells visited by it
        graph: dual graph of the mesh
        mesh: original mesh
        node2tris: a mapping from a mesh node to all triangles incident to it

    Return:

    '''

    # initialization
    num_cells = len(mesh.faces)
    num_nodes = len(mesh.vertices)
    visited_triangles = np.zeros(num_cells)
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
        unvisited_triangles = np.where(visited_triangles == 0)[0]
        seed = np.random.choice(unvisited_triangles, 1)[0]
        for n in mesh.faces[seed]:
            visited_triangles, visited_nodes = visitedUpdate(n, visited_triangles, visited_nodes, mesh, node2tris)
            decomp_func[n] = func[n]
        compressed_code_end = compressed_code_start + int(sequence_lengths[seed_idx])
        visited_triangles, visited_nodes, decomp_func = \
            DFS_decompression_oneSeed(compressed_code[compressed_code_start:compressed_code_end], xi, graph, mesh,
                                      node2tris, seed, visited_triangles, visited_nodes, decomp_func)
        seed_idx += 1
        compressed_code_start = compressed_code_end

    return decomp_func


def DFS_decompression_oneSeed(compressed: list, xi: float, graph: nx.Graph, mesh: trimesh.Trimesh, node2tris: dict,
                              seed: int, visited_triangles: np.array, visited_nodes: np.array, decomp_func: np.array):
    '''
    This function conducts DFS and decompression for ONE SEED on dual graph of a triangular mesh with two conditions:
    1) Early stop: if the newly interpolated mesh node is unpredictable;
    2) Visited constraint: if all three mesh nodes of a triangle is interpolated, mark the triangle as visited.

    Augments:
        compressed: a list of quantization codes following the seed
        graph: dual graph of the mesh
        mesh: original mesh
        node2tris: a mapping from a mesh node to all triangles incident to it
        seed: index of a seed triangle
        visited_triangles: if triangles are visited
        visited_nodes: if nodes are visited
        decomp_func: decomposed function values on mesh nodes

    Return:

    '''
    stack = [(seed, None)]
    stack_set = {(seed, None)}
    while len(compressed) > 0:
        current_tri, prev_tri = stack.pop()
        stack_set.remove((current_tri, prev_tri))
        if visited_triangles[current_tri] == 0:  # visited constraint
            node_to_update = list(set(mesh.faces[current_tri]) - set(mesh.faces[prev_tri]))[0]
            pred_f = triangleInterpolation(mesh.vertices[node_to_update], prev_tri, mesh, decomp_func)
            code = compressed[0]
            compressed = np.delete(compressed, 0)
            code_distance = code - 2 ** (m - 1)
            visited_triangles, visited_nodes = visitedUpdate(node_to_update, visited_triangles, visited_nodes, mesh,
                                                             node2tris)
            decomp_func[node_to_update] = pred_f + code_distance * 2 * xi
        neighbors = list(graph.neighbors(current_tri))
        neighbors.sort(reverse=True)  # search strategy: start with cell with min index
        for t in neighbors:
            if visited_triangles[t] == 0 and t not in stack_set:
                stack.append((t, current_tri))
                stack_set.add((t, current_tri))

    return visited_triangles, visited_nodes, decomp_func


if __name__ == "__main__":
    f = nc.Dataset("output.nc")
    nodes_coor = np.concatenate((np.array(f.variables["xCell"]).reshape(-1, 1),
                                 np.array(f.variables["yCell"]).reshape(-1, 1),
                                 np.array(f.variables["zCell"]).reshape(-1, 1)), axis=1)
    cells = np.array(f.variables['cellsOnVertex'])
    reindex = np.where(cells == 0)
    for i in range(reindex[0].shape[0]):
        cells[reindex[0][i]][reindex[1][i]] = 1
    cells -= 1
    func = np.array(f.variables["temperature"])[0, :, 0]
    num_nodes = nodes_coor.shape[0]
    num_cells = cells.shape[0]
    dual_graph = dualGraph4MeshTriCells(cells, num_cells)
    mesh = trimesh.Trimesh(vertices=nodes_coor, faces=cells)
    node2tris = node2trisDict(cells, num_nodes)

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
        sequences, compressed_codes, cr, codec = DFS_compression(dual_graph, xi, mesh, node2tris, func)
        t1 = time.perf_counter()
        runtime[i, 0] = t1 - t0
        t0 = time.perf_counter()
        decompressed_vals = DFS_decompression(compressed_codes, xi, codec, dual_graph, mesh, node2tris)
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
    axs[0, 2].set_ylabel("MSE")
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
    fig.suptitle("MPAS-ocean data")
    plt.savefig("mpas-ocean.png")

    eval_metrics = {"runtime": runtime, "nrmse": nrmse, "psnr": psnr, "cr": compression_ratio, "br": bit_rate}
    with open("eval_metrics.pickle", "wb") as f:
        pickle.dump(eval_metrics, f)
