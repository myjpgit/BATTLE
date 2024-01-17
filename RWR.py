import numpy as np
import random
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch

def gdc(A, alpha, eps=0.01):
    A = csr_matrix(A)
    N = A.shape[0]
    # Self-loops
    A_loop = sp.eye(N) + A
    #Symmetric transition matrix
    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsgrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = D_loop_invsgrt @ A_loop @ D_loop_invsgrt
    # PPR-based diffusion
    S = alpha * sp.linalg.inv(sp.eye(N) - (1-alpha) * T_sym)
    # Sparsify using threshold epsilon
    S_tilde = S.multiply(S >= eps)
    # Column-normalized transition matrix on graph S tilde
    D_tilde_vec = S_tilde.sum(0).A1
    T_S = S_tilde / D_tilde_vec
    return T_S

def random_walk_with_restart(adj_matrix, restart_prob, num_steps):
    num_nodes = adj_matrix.shape[0]
    subgraph_matrix = np.zeros_like(adj_matrix)

    for start_node in range(num_nodes):
        current_node = start_node
        subgraph = [current_node]

        for _ in range(num_steps):
            if random.random() < restart_prob:
                current_node = start_node
            else:
                neighbors = np.nonzero(adj_matrix[current_node])[0]
                if len(neighbors) > 0:
                    current_node = random.choice(neighbors)
                else:
                    break
            subgraph.append(current_node)

        subgraph_matrix[start_node, subgraph] = 1

    return subgraph_matrix


# # 示例邻接矩阵
# adj_matrix = np.array([[0, 1, 1, 0],
#                        [1, 0, 1, 1],
#                        [1, 1, 0, 1],
#                        [0, 1, 1, 0]])

# restart_prob = 0.5  # 重启概率
# num_steps = 5 # 游走步数

# subgraph_matrix = random_walk_with_restart(adj_matrix, restart_prob, num_steps)
# print("采样的邻接矩阵:\n", subgraph_matrix)