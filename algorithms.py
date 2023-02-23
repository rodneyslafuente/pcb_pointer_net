import numpy as np
import torch

from typing import Tuple


def greedily_decode_pointers(pointers: torch.Tensor) -> np.ndarray:
    """ 
    Greedily choose a valid route from pointers.

    :param Tensor pointers: Pointers (batch_size, num_nodes, num_nodes)
    :return: Valid routes (batch_size, num_nodes)
    """

    pred = []
    for pointer in pointers:
        i = torch.argmax(pointer)

        while i in pred:
            pointer = torch.cat(
                [pointer[:i], torch.tensor([float("-inf")]), pointer[i + 1 :]]
            )
            i = torch.argmax(pointer)

        pred.append(i)

    return torch.tensor(pred).detach().numpy()


def make_dist_matrix(positions: np.ndarray) -> np.ndarray:
    """ 
    Create an adjacency matrix `dist_mat` given a set of coordinates. `dist_mat[i][j]` gives the 
    euclidean distance between coordinates `i` and `j`.

    :param ndarry positions: Set of coordinates (num_nodes, 2)
    :return: Distance matrix (num_nodes, num_nodes)
    """

    def euclidean_dist(pos_1, pos_2):
        return ((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2) ** 0.5

    N = len(positions)
    dists = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            dists[i][j] = dists[j][i] = euclidean_dist(positions[i], positions[j])

    return dists


def cost(route: np.ndarray, dist_mat: np.ndarray) -> float:
    """
    Compute the length of a single route given a distance matrix

    :input ndarray route: (num_nodes, )
    :input ndarray dist_mat: (num_nodes, num_nodes)
    :return: The length, or cost, of the given route as a float
    """

    return dist_mat[np.roll(route, 1).astype(int), route].sum()


def two_opt(route: np.ndarray, dist_mat: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Find the locally minimum route given a starting route by iteratively swapping
    pairs of nodes if doing so decreases the route length until no improvement is made

    :input ndarray route: (num_nodes, )
    :input ndarray dist_mat: (num_nodes, num_nodes)
    :return: 
    """

    best = np.copy(route)
    improved = True

    while improved:
        improved = False

        for i in range(1, len(route) - 2):
            for j in range(i + 2, len(route)):
                new_route = np.copy(route)
                new_route[i:j] = route[j - 1 : i - 1 : -1]  # swap to create new route

                if cost(new_route, dist_mat) < cost(best, dist_mat):
                    best = new_route
                    improved = True

        route = best

    return best, cost(best, dist_mat)
