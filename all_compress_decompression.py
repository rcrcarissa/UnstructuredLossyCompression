import h5py
import numpy as np
import networkx as nx
import trimesh
import struct
import pickle
import zlib
import time
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import sys

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


def DFS_compression(graph: nx.Graph, mesh: trimesh.Trimesh, node2tris: dict, func: np.array):
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
        a dictionary of compressed sequences in following format:
        {seed_triangle_id0: [quan_code00  quan_code01  ...],
        seed_triangle_id1: [quan_code10  quan_code11  ...],
        ...}
    '''

    # initialization
    num_cells = len(mesh.faces)
    num_nodes = len(mesh.vertices)
    visited_triangles = np.zeros(num_cells)
    visited_nodes = np.zeros(num_nodes)
    decomp_func = np.zeros(num_nodes)
    np.random.seed(seed_instance)
    sequences = {}
    compressed_codes = {}

    while not all(visited_nodes):
        unvisited_triangles = np.where(visited_triangles == 0)[0]
        seed = np.random.choice(unvisited_triangles, 1)[0]
        for n in mesh.faces[seed]:
            visited_triangles, visited_nodes = visitedUpdate(n, visited_triangles, visited_nodes, mesh, node2tris)
            decomp_func[n] = func[n]
        sequence, compressed_code, visited_triangles, visited_nodes, decomp_func = \
            DFS_compression_oneSeed(graph, mesh, node2tris, seed, visited_triangles, visited_nodes, func, decomp_func)
        sequences[seed] = sequence
        compressed_codes[seed] = compressed_code

    return sequences, compressed_codes, visited_triangles, visited_nodes


def DFS_compression_oneSeed(graph: nx.Graph, mesh: trimesh.Trimesh, node2tris: dict, seed: int,
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
        compressed_code: a compressed sequences in list format
    '''
    stack = [(seed, None)]
    sequence = []
    compressed_code = []
    while len(stack) > 0:
        current_tri, prev_tri = stack.pop()
        if visited_triangles[current_tri] == 0:  # visited constraint
            node_to_update = list(set(mesh.faces[current_tri]) - set(mesh.faces[prev_tri]))[0]
            pred_f = triangleInterpolation(mesh.vertices[node_to_update], prev_tri, mesh, decomp_func)
            code_distance = int((func[node_to_update] - pred_f) / xi)
            if code_distance < 0:
                code = 2 ** (m - 1) + int(np.floor(code_distance / 2))
            else:
                code = 2 ** (m - 1) + int(np.ceil(code_distance / 2))
            if code < 1 or code > 2 ** m - 1:
                break  # unpredictable; early stop
            else:
                decomp_func[node_to_update] = pred_f + (code - 2 ** (m - 1)) * 2 * xi
                code = int(bin(code)[2:])
                sequence.append(current_tri)
                compressed_code.append(code)
                visited_triangles, visited_nodes = visitedUpdate(node_to_update, visited_triangles, visited_nodes, mesh,
                                                                 node2tris)
        neighbors = list(graph.neighbors(current_tri))
        neighbors.sort(reverse=True)  # search strategy: start with cell with min index
        for t in neighbors:
            if visited_triangles[t] == 0 and t not in set(stack):
                stack.append((t, current_tri))
    return sequence, compressed_code, visited_triangles, visited_nodes, decomp_func


def DFS_decompression(compressed_codes: dict, graph: nx.Graph, mesh: trimesh.Trimesh, node2tris: dict):
    '''
    This function conducts DFS and decompression on dual graph of a triangular mesh with two conditions:
    1) Early stop: if the newly interpolated mesh node is unpredictable;
    2) Visited constraint: if all three mesh nodes of a triangle is interpolated, mark the triangle as visited.

    Augments:
        compressed_codes: a dictionary of compressed sequences in following format:
        {seed_triangle_id0: [quan_code00  quan_code01  ...],
        seed_triangle_id1: [quan_code10  quan_code11  ...],
        ...}
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
    i = 0

    for seed, compressed_code in compressed_codes.items():
        for n in mesh.faces[seed]:
            visited_triangles, visited_nodes = visitedUpdate(n, visited_triangles, visited_nodes, mesh, node2tris)
            decomp_func[n] = func[n]
        visited_triangles, visited_nodes, decomp_func, i = \
            DFS_decompression_oneSeed(compressed_code, graph, mesh, node2tris, seed, visited_triangles, visited_nodes,
                                      decomp_func, i)

    return decomp_func, visited_triangles, visited_nodes


def DFS_decompression_oneSeed(compressed_code: list, graph: nx.Graph, mesh: trimesh.Trimesh, node2tris: dict, seed: int,
                              visited_triangles: np.array, visited_nodes: np.array, decomp_func: np.array, i: int):
    '''
    This function conducts DFS and decompression for ONE SEED on dual graph of a triangular mesh with two conditions:
    1) Early stop: if the newly interpolated mesh node is unpredictable;
    2) Visited constraint: if all three mesh nodes of a triangle is interpolated, mark the triangle as visited.

    Augments:
        compressed_code: a list of quantization codes following the seed
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
    idx = 0
    while idx < len(compressed_code):
        current_tri, prev_tri = stack.pop()
        if visited_triangles[current_tri] == 0:  # visited constraint
            node_to_update = list(set(mesh.faces[current_tri]) - set(mesh.faces[prev_tri]))[0]
            pred_f = triangleInterpolation(mesh.vertices[node_to_update], prev_tri, mesh, decomp_func)
            code = compressed_code[idx]
            code_distance = int('0b' + str(code), 2) - 2 ** (m - 1)
            visited_triangles, visited_nodes = visitedUpdate(node_to_update, visited_triangles, visited_nodes, mesh,
                                                             node2tris)
            decomp_func[node_to_update] = pred_f + code_distance * 2 * xi
            idx += 1
        neighbors = list(graph.neighbors(current_tri))
        neighbors.sort(reverse=True)  # search strategy: start with cell with min index
        for t in neighbors:
            if visited_triangles[t] == 0 and t not in set(stack):
                stack.append((t, current_tri))
        i += 1
        # drawVisited(visited_triangles, visited_nodes, mesh, i)
    return visited_triangles, visited_nodes, decomp_func, i


if __name__ == "__main__":
    with h5py.File("../../../TimeVaryingMesh/xgc/xgc.mesh.h5", "r") as f:
        cells = np.array(f["cell_set[0]"]["node_connect_list"])
        nodes_coor = np.array(f["coordinates"]["values"])
        num_cells = f["n_t"][0]
        num_nodes = f["n_n"][0]
    with h5py.File("../../../TimeVaryingMesh/xgc/xgc.3d.00165.h5", "r") as f:
        func = np.array(f["dneOverne0"])[:, 0]
    dual_graph = dualGraph4MeshTriCells(cells, num_cells)
    mesh = trimesh.Trimesh(vertices=np.concatenate((nodes_coor, np.zeros((num_nodes, 1))), axis=1), faces=cells)
    node2tris = node2trisDict(cells, num_nodes)

    file = ''
    for n in range(num_nodes):
        file += str(func[n]) + ' '
    file = file[:-1]

    max_signal = np.max(func)
    xi_dim = len(xi_range)
    runtime = np.zeros((xi_dim, 2))
    mse = np.zeros(xi_dim)
    psnr = np.zeros(xi_dim)
    compression_ratio = np.zeros(xi_dim)
    bit_rate = np.zeros(xi_dim)

    for i, xi in enumerate(xi_range):
        t0 = time.perf_counter()
        sequences, compressed_codes, _, _ = DFS_compression(dual_graph, mesh, node2tris, func)
        t1 = time.perf_counter()
        runtime[i, 0] = t1 - t0
        print(t1 - t0)
        t0 = time.perf_counter()
        decompressed_vals, _, _ = DFS_decompression(compressed_codes, dual_graph, mesh, node2tris)
        t1 = time.perf_counter()
        runtime[i, 1] = t1 - t0
        print(t1 - t0)
        mse[i] = np.sum((decompressed_vals - func) ** 2) / num_nodes
        psnr[i] = 20 * np.log10(max_signal) - 10 * np.log10(mse[i])

        to_comp = bytearray(b'')
        for seed, sequence in compressed_codes.items():
            to_comp += int(seed).to_bytes(m, 'big')
            for n in cells[seed]:
                to_comp += struct.pack("f", func[n])
            for v in sequence:
                integer = int('0b' + str(v), 2)
                to_comp += integer.to_bytes(m, 'big')
        comp = zlib.compress(to_comp)
        compression_ratio[i] = sys.getsizeof(file) / sys.getsizeof(comp)
        bit_rate[i] = 64 / compression_ratio[i]

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
    axs[0, 2].plot(xi_range, mse)
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
    axs[1, 2].plot(bit_rate, mse)
    axs[1, 2].set_xlabel("bit rate")
    axs[1, 2].set_ylabel("MSE")
    axs[1, 3].plot(bit_rate, psnr)
    axs[1, 3].set_xlabel("bit rate")
    axs[1, 3].set_ylabel("PSNR (dB)")
    fig.suptitle("XGC data")
    plt.savefig("xgc.png")

    print(runtime)
    print(mse)
    print(psnr)
    print(compression_ratio)
    print(bit_rate)
