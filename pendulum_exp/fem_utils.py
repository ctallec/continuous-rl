""" Cylindrical mesh generation. """
from typing import List, Tuple
import numpy as np
from skfem import Mesh, MeshTri

def find_neighbours(vertices: np.ndarray,
                    left_boundary_indices: np.ndarray,
                    right_boundary_indices: np.ndarray,
                    delta: float):
    """ Find neighbours. """
    left_boundary_vertices = vertices[:, left_boundary_indices]
    right_boundary_vertices = vertices[:, right_boundary_indices]
    neighbours = []
    for i, l_vertex in enumerate(left_boundary_vertices[1, :]):
        l_neighbours = []
        for j, r_vertex in enumerate(right_boundary_vertices[1, :]):
            if np.abs(l_vertex - r_vertex) < delta / 3:
                l_neighbours.append([left_boundary_indices[i],
                                     right_boundary_indices[j]])
            if np.abs(l_vertex + delta - r_vertex) < delta / 3:
                l_neighbours.append([left_boundary_indices[i],
                                     right_boundary_indices[j]])
        neighbours.append(l_neighbours)

    for i, r_vertex in enumerate(right_boundary_vertices[1, :]):
        r_neighbours = []
        for j, l_vertex in enumerate(left_boundary_vertices[1, :]):
            if np.abs(l_vertex - r_vertex) < delta / 3:
                r_neighbours.append([right_boundary_indices[i],
                                     left_boundary_indices[j]])
            if np.abs(l_vertex + delta - r_vertex) < delta / 3:
                r_neighbours.append([right_boundary_indices[i],
                                     left_boundary_indices[j]])
        neighbours.append(r_neighbours)
    result: List[Tuple[int, int, int]] = []
    for l in neighbours:
        if len(l) == 2:
            result.append((l[0][0], l[0][1], l[1][1]))
    return np.array(result).transpose(1, 0)

def cylindrical_mesh(scales: Tuple[float, float],
                     translation: Tuple[float, float],
                     resolution: int) -> Mesh:
    """ Returns a cylindrical mesh. """
    s_x, s_y = scales
    eps = 1e-5
    delta = 1 / 2 ** resolution
    max_x = s_x - delta
    m = MeshTri()
    m.refine(resolution)
    m.scale((max_x, s_y))
    m.translate(translation)

    max_x = max_x + translation[0]
    min_x = translation[0]

    cyclic_p = m.p.copy()
    cyclic_t = m.t.copy()
    right_boundary_indices = [i for i, c in enumerate(cyclic_p[0])
                              if np.abs(c - max_x) < eps]
    left_boundary_indices = [i for i, c in enumerate(cyclic_p[0])
                             if np.abs(c - min_x) < eps]

    new_neighbours = find_neighbours(
        cyclic_p, left_boundary_indices, right_boundary_indices, delta * s_y)

    cyclic_t = np.concatenate((cyclic_t, new_neighbours), axis=1)
    m = MeshTri(cyclic_p, cyclic_t)
    return m

if __name__ == '__main__':
    m = cylindrical_mesh((2 * np.pi, 10), (-np.pi, -5), 5)
    m.draw()
    input()
