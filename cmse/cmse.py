import numpy as np
import pickle


def triArea(tri):
    return np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) / 2


def tetVolume(tet):
    return np.dot(np.cross(tet[1] - tet[0], tet[2] - tet[0]), tet[3] - tet[0]) / 6


def triCellwiseError(tri, f0, f1):
    J = np.array([[tri[0][0] - tri[2][0], tri[1][0] - tri[2][0]], [tri[0][1] - tri[2][1], tri[1][1] - tri[2][1]]])
    J = np.abs(np.linalg.det(J))
    e = 0
    for i in range(3):
        for j in range(i, 3):
            e += (f1[i] - f0[i]) * (f1[j] - f0[j])
    return J * e / 12


def tetCellwiseError(tet, f0, f1):
    J = np.array([[tet[0][0] - tet[3][0], tet[1][0] - tet[3][0], tet[2][0] - tet[3][0]],
                  [tet[0][1] - tet[3][1], tet[1][1] - tet[3][1], tet[2][1] - tet[3][1]],
                  [tet[0][2] - tet[3][2], tet[1][2] - tet[3][2], tet[2][2] - tet[3][2]]])
    J = np.abs(np.linalg.det(J))
    e = 0
    for i in range(4):
        for j in range(i, 4):
            e += (f1[i] - f0[i]) * (f1[j] - f0[j])
    return J * e / 60


def triangularMeshCMSE(nodes_coor, cells, func0, func1):
    num_cells = cells.shape[0]
    nodes_dim = nodes_coor.shape[1]
    coors_p0, coors_p1, coors_p2 = np.zeros((num_cells, 3)), np.zeros((num_cells, 3)), np.zeros((num_cells, 3))
    coors_p01 = np.zeros((num_cells, 2, 2))

    if nodes_dim == 3:
        coors_tmp00 = nodes_coor[cells[:, 1]][:, 0] - nodes_coor[cells[:, 0]][:, 0]
        coors_tmp01 = nodes_coor[cells[:, 1]][:, 1] - nodes_coor[cells[:, 0]][:, 1]
        coors_tmp02 = nodes_coor[cells[:, 1]][:, 2] - nodes_coor[cells[:, 0]][:, 2]
        coors_tmp10 = nodes_coor[cells[:, 2]][:, 0] - nodes_coor[cells[:, 0]][:, 0]
        coors_tmp11 = nodes_coor[cells[:, 2]][:, 1] - nodes_coor[cells[:, 0]][:, 1]
        coors_tmp12 = nodes_coor[cells[:, 2]][:, 2] - nodes_coor[cells[:, 0]][:, 2]
        coors_p1[:, 0] = np.sqrt(coors_tmp00 ** 2 + coors_tmp01 ** 2 + coors_tmp02 ** 2)
        coors_p2[:, 0] = \
            (coors_tmp00 * coors_tmp10 + coors_tmp01 * coors_tmp11 + coors_tmp02 * coors_tmp12) / coors_p1[:, 0]
        coors_p2[:, 1] = np.sqrt(coors_tmp10 ** 2 + coors_tmp11 ** 2 + coors_tmp12 ** 2 - coors_p2[:, 0] ** 2)
    else:
        coors_p0[:, :2] = nodes_coor[cells[:, 0]]
        coors_p1[:, :2] = nodes_coor[cells[:, 1]]
        coors_p2[:, :2] = nodes_coor[cells[:, 2]]
    coors_p01[:, 0, :] = coors_p0[:, :2]
    coors_p01[:, 1, :] = coors_p1[:, :2]
    total_area = np.cross(coors_p1 - coors_p0, coors_p2 - coors_p0)
    total_area = np.linalg.norm(total_area, axis=1)
    total_area = total_area.sum() / 2

    J = coors_p01 - np.repeat(coors_p2[:, :2].reshape(-1, 1, 2), 2, axis=1)
    J = np.abs(np.linalg.det(J))
    f0 = func1[cells[:, 0]] - func0[cells[:, 0]]
    f1 = func1[cells[:, 1]] - func0[cells[:, 1]]
    f2 = func1[cells[:, 2]] - func0[cells[:, 2]]
    e = f0 ** 2 + f0 * f1 + f0 * f2 + f1 ** 2 + f1 * f2 + f2 ** 2
    total_cellwsie_error = (J * e).sum() / 12

    return total_cellwsie_error / total_area


def tetraheronMeshCMSE(nodes_coor, cells, func0, func1):
    num_cells = cells.shape[0]
    mats = np.zeros((num_cells, 3, 3))
    coors_p0 = nodes_coor[cells[:, 0]]
    coors_p3 = nodes_coor[cells[:, 3]]
    mats[:, 0, :] = nodes_coor[cells[:, 1]] - coors_p0
    mats[:, 1, :] = nodes_coor[cells[:, 2]] - coors_p0
    mats[:, 2, :] = coors_p3 - coors_p0
    total_volume = np.sum(np.abs(np.linalg.det(mats))) / 6

    J = nodes_coor[cells[:, :3]] - np.repeat(coors_p3.reshape(-1, 1, 3), 3, axis=1)
    J = np.abs(np.linalg.det(J))
    f0 = func1[cells[:, 0]] - func0[cells[:, 0]]
    f1 = func1[cells[:, 1]] - func0[cells[:, 1]]
    f2 = func1[cells[:, 2]] - func0[cells[:, 2]]
    f3 = func1[cells[:, 3]] - func0[cells[:, 3]]
    e = f0 ** 2 + f0 * f1 + f0 * f2 + f0 * f3 + f1 ** 2 + f1 * f2 + f1 * f3 + f2 ** 2 + f2 * f3 + f3 ** 2
    total_cellwsie_error = (J * e).sum() / 60

    return total_cellwsie_error / total_volume
