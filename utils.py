#!/usr/bin/env python
# -*- coding:utf8 -*-

import numpy as np
import time
import scipy.sparse as ssp
import random
import math
from torch_geometric.data import (InMemoryDataset, Data)

import torch
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import degree

import multiprocessing as mp
from sklearn import metrics
import networkx as nx
from tqdm import tqdm
from torch_geometric.transforms import LineGraph


def construct_matrix_test(data):
    node_nums = data.x.shape[0]
    adj = to_dense_adj(data.edge_index, max_num_nodes=node_nums)
    hop_2_adj = torch.bmm(adj, adj) + adj
    hop_2_adj[hop_2_adj == 0] = 999
    hop_2_adj[hop_2_adj < 999] = 0
    hop_2_adj[hop_2_adj == 999] = 1
    del adj

    return hop_2_adj


def convert_to_one_hot_label(labels, num_classes, batch_size, device):
    one_hot_y = labels.unsqueeze(1).to(device)
    n_class = num_classes
    batch_size = batch_size
    y_onehot = torch.FloatTensor(batch_size, n_class).to(device)
    y_onehot.zero_()
    y_onehot = y_onehot.scatter(1, one_hot_y, 1)
    return y_onehot


class ProcessedDataset(InMemoryDataset):
    pass


def precompute_edge_label_and_reverse(dataset: InMemoryDataset):
    data_list = []
    for data in dataset:
        u, v = data.edge_index
        yu, yv = data.y[u], data.y[v]
        data.edge_labels = yu * dataset.num_classes + yv

        edge_dict = torch.sparse_coo_tensor(indices=data.edge_index,
                                            values=torch.arange(data.num_edges),
                                            size=(data.num_nodes,
                                                  data.num_nodes)).to_dense()
        data.edge_index_reversed = edge_dict[v, u]

        data_list.append(data)

    new_data, new_slices = InMemoryDataset.collate(data_list)
    new_dataset = ProcessedDataset('.')
    new_dataset.data = new_data
    new_dataset.slices = new_slices
    return new_dataset


class CitationDataset(InMemoryDataset):
    def __init__(self, root=None, split='train', transform=None,
                 pre_transform=None, pre_filter=None):
        assert split in ['train', 'val', 'test']
        super(CitationDataset, self).__init__(root, transform, pre_transform,
                                              pre_filter)

        saved_data = torch.load(root)

        self.data = Data(edge_index=saved_data['{}_e'.format(split)],
                         x=saved_data['{}_x'.format(split)],
                         y=saved_data['{}_y'.format(split)])
        num_nodes = self.data.x.size(0)
        num_edges = self.data.edge_index.size(1)
        self.slices = {
            'x': torch.LongTensor([0, num_nodes]),
            'y': torch.LongTensor([0, num_nodes]),
            'edge_index': torch.LongTensor([0, num_edges])
        }
        if self.pre_transform is not None:
            self.data = self.pre_transform(self.data)


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def find_common_neighbors_matrix(batch_data,
                                 num_nodes,
                                 device,
                                 model_name=None):
    if model_name == 'gcnii':
        row, col, edge_attr = batch_data.adj_t.t().coo()
        edge_index = torch.stack([row, col], dim=0)
        batch_data.edge_index = edge_index
    adj = to_dense_adj(batch_data.edge_index,
                       max_num_nodes=num_nodes)[0]
    adj1 = torch.index_select(adj,
                              0,
                              batch_data.n_id[:batch_data.batch_size]).to(
        device)
    adj2 = torch.index_select(adj1.T,
                              0,
                              batch_data.n_id[:batch_data.batch_size]).to(
        device)
    common_neighbors_tensor = []
    for i in range(adj1.shape[0]):
        common_neighbors_tensor.append(
            torch.sum(torch.logical_and(adj2[i], adj2), dim=1))

    common_neighbors_tensor = torch.stack(common_neighbors_tensor)
    mask = torch.eye(batch_data.batch_size,
                     batch_data.batch_size).byte().to(device)
    common_neighbors_tensor.masked_fill_(mask, 0)

    return common_neighbors_tensor


def count_degree(batch_data,
                 num_nodes,
                 device,
                 model_name=None):
    if model_name == 'gcnii':
        row, col, edge_attr = batch_data.adj_t.t().coo()
        edge_index = torch.stack([row, col], dim=0)
        batch_data.edge_index = edge_index

    degree_vec = degree(batch_data.edge_index[1], num_nodes=num_nodes) + degree(
        batch_data.edge_index[0], num_nodes=num_nodes)
    degree_res = []

    for i in range(degree_vec.shape[0]):
        degree_res.append(degree_vec - degree_vec[i])

    degree_res = -torch.abs(torch.stack(degree_res))
    degree_res.fill_diagonal_(-999)

    adj1 = torch.index_select(degree_res,
                              0,
                              batch_data.n_id[:batch_data.batch_size]).to(
        device)
    adj2 = torch.index_select(adj1.T,
                              0,
                              batch_data.n_id[:batch_data.batch_size]).to(
        device)

    return adj2
