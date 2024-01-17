import warnings
import pickle as pkl
import sys, os
import networkx as nx
import torch
import numpy as np

import scipy.sparse as sp
import networkx as nx
import torch
import numpy as np
import os.path as osp
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
# from utils import get_link_labels, evaluate_AUC, link_prediction, generate_pos_edge_index, generate_neg_edge_index
from collections import defaultdict

# from sklearn import datasets
# from sklearn.preprocessing import LabelBinarizer, scale
# from sklearn.model_selection import train_test_split
# from ogb.nodeproppred import DglNodePropPredDataset
# import copy

from utils import sparse_mx_to_torch_sparse_tensor #, dgl_graph_to_torch_sparse

warnings.simplefilter("ignore")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_citation_network(dataset_str, sparse=None, num_val=0.1, num_test=0.3):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset_str)
    if dataset_str in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(path, dataset_str)
    elif dataset_str in ["BlogCatalog", "Flickr", "ACM", "Advisor", "Samecity", 'Cora']:
        dataset = TUDataset(path, dataset_str, T.NormalizeFeatures(), use_node_attr=True, use_edge_attr=True)

    data = dataset[0]

    features = data.x  # 得到整个数据的全部特征矩阵
    num_train = 1 - num_val - num_test
    train_num = int(data.num_nodes*num_train)
    val_num = int(data.num_nodes*num_val)
    test_num = int(data.num_nodes*num_test)
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[:train_num] = 1
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask[train_num: train_num+val_num] = 1
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[train_num+val_num:] = 1

    # data.pos_edge_index = generate_pos_edge_index(data.edge_index, data.y)
    # graph = generate_data(data.pos_edge_index, data.num_nodes)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(data.graph)) # 得到邻接矩阵
    if not sparse:
        adj = np.array(adj.todense(),dtype='float32')
    else:
        adj = sparse_mx_to_torch_sparse_tensor(adj)

    labels = torch.tensor(data.y , dtype=torch.int64)  # 得到整个数据的全部标签
    edge_index = data.edge_index
    labels = torch.LongTensor(labels) # 将label转换为torch.Long

    nfeats = features.shape[1] # 特征维度
    nclasses = torch.max(labels).item() + 1 # labels中数字最大的值对应节点总共的类别数

    train_mask = torch.BoolTensor(data.train_mask)
    val_mask = torch.BoolTensor(data.val_mask)
    test_mask = torch.BoolTensor(data.test_mask)

    return data, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj, edge_index


    # names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    # objects = []
    # for i in range(len(names)):
    #     with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
    #         if sys.version_info > (3, 0):
    #             objects.append(pkl.load(f, encoding='latin1'))
    #         else:
    #             objects.append(pkl.load(f))

    # x, y, tx, ty, allx, ally, graph = tuple(objects)
    # test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    # test_idx_range = np.sort(test_idx_reorder)

    # if dataset_str == 'citeseer':
    #     # Fix citeseer dataset (there are some isolated nodes in the graph)
    #     # Find isolated nodes, add them as zero-vecs into the right position
    #     test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
    #     tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    #     tx_extended[test_idx_range - min(test_idx_range), :] = tx
    #     tx = tx_extended
    #     ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    #     ty_extended[test_idx_range - min(test_idx_range), :] = ty
    #     ty = ty_extended

    # features = sp.vstack((allx, tx)).tolil()
    # features[test_idx_reorder, :] = features[test_idx_range, :]

    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # if not sparse:
    #     adj = np.array(adj.todense(),dtype='float32')
    # else:
    #     adj = sparse_mx_to_torch_sparse_tensor(adj)

    # labels = np.vstack((ally, ty))
    # labels[test_idx_reorder, :] = labels[test_idx_range, :]
    # idx_test = test_idx_range.tolist()
    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y) + 500)

    # train_mask = sample_mask(idx_train, labels.shape[0])
    # val_mask = sample_mask(idx_val, labels.shape[0])
    # test_mask = sample_mask(idx_test, labels.shape[0])

    # features = torch.FloatTensor(features.todense())
    # labels = torch.LongTensor(labels)
    # train_mask = torch.BoolTensor(train_mask)
    # val_mask = torch.BoolTensor(val_mask)
    # test_mask = torch.BoolTensor(test_mask)

    # nfeats = features.shape[1]
    # for i in range(labels.shape[0]):
    #     sum_ = torch.sum(labels[i])
    #     if sum_ != 1:
    #         labels[i] = torch.tensor([1, 0, 0, 0, 0, 0])
    # labels = (labels == 1).nonzero()[:, 1]
    # nclasses = torch.max(labels).item() + 1

    # return features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj


def load_data(args, num_val=0.2, num_test=0.3):
    return load_citation_network(args.dataset, args.sparse, num_val=0.2, num_test=0.3)