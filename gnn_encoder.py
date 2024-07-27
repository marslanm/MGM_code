
import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, SGConv, GCN2Conv, GATv2Conv
from torch_geometric.nn import FiLMConv
from torch_geometric.nn import DNAConv
#from torch_geometric.nn import SSGConv
from torch_geometric.nn import FAConv
from utils import convert_to_one_hot_label, sim_matrix
import math
from tqdm import tqdm
from torch.nn import BatchNorm1d

class GraphSAGE_Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 criterion,
                 k,
                 eta,
                 device):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layer_1 = SAGEConv(input_dim, hidden_dim)
        self.layer_2 = SAGEConv(hidden_dim, num_classes)
        self.criterion = criterion
        self.k = k
        self.eta = eta
        if criterion == 'sigmoid':
            self.m = nn.Sigmoid()
        elif criterion == 'softmax':
            self.m = nn.LogSoftmax(dim=1)
        self.num_classes = num_classes
        self.device = device


    def pretrain_with_batch(self, data):

        x, edge_index = data.x, data.edge_index
        x = self.layer_1(x, edge_index)
        x = x.relu_()
        x = F.dropout(x, p=0.5, training=self.training)
        lc_x = self.layer_2(x, edge_index)
        lc_x = lc_x[:data.batch_size]
        p_lc = self.m(lc_x)

        return p_lc


    @torch.no_grad()
    def get_embedding(self, data):
        dic = {}
        dic['full_data'] = data
        x, edge_index = data.x, data.edge_index
        #y = data.y[:data.batch_size]
        y = data.y[data.train_mask][:data.batch_size]
        x = self.layer_1(x, edge_index)
        x = x.relu_()
        #embedding = x[:data.batch_size]
        embedding = x[data.train_mask][:data.batch_size]

        dic['train_embedding'] = embedding
        dic['train_label'] = y

        return dic


    @torch.no_grad()
    def decode_z_q(self, train_info):

        data = train_info['full_data']
        x, edge_index = data.x, data.edge_index

        y = data.y[:data.batch_size]
        x = self.layer_1(x, edge_index)
        x = x.relu_()
        x = F.dropout(x, p=0.5, training=self.training)

        lc_x = self.layer_2(x, edge_index)
        #lc_x = lc_x[:data.batch_size]
        lc_x = lc_x[data.train_mask][:data.batch_size]
        p_lc = self.m(lc_x)
        # breakpoint()
        p_sim = self.m(self.generate_label_with_similar_nodes(train_info['z_q'],
                                                              train_info['z_q'],
                                                              #y,
                                                              data.y[data.train_mask],
                                                              train_info['neighbors_info']))
        #breakpoint()
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution


    def forward(self,
                data,
                neighbors_info=None):
        x, edge_index = data.x, data.edge_index

        y = data.y[:data.batch_size]
        x = self.layer_1(x, edge_index)
        # x = F.elu(x)
        x = x.relu_()
        x = F.dropout(x, p=0.5, training=self.training)
        #embedding = x[:data.batch_size]
        embedding = x[data.train_mask][:data.batch_size]

        lc_x = self.layer_2(x, edge_index)
        lc_x = lc_x[data.train_mask][:data.batch_size]
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                              embedding,
                                                              #y,
                                                              data.y[data.train_mask],
                                                              neighbors_info))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution, embedding


    def generate_label_with_similar_nodes(self,
                                          embedding_a,
                                          embedding_b,
                                          b_y,
                                          neighbors_info=None):
        if self.training:
            mask = torch.ones(b_y.shape[0], b_y.shape[0])
            torch.diagonal(mask, 0).zero_()
        with torch.no_grad():
            if torch.is_tensor(neighbors_info):
                # print("|> [generate_label_with_similar_nodes] Using neighbors information")
                feature_similarity_matrix = neighbors_info
            else:
                # print("|> [generate_label_with_similar_nodes] Not Using neighbors information")
                feature_similarity_matrix = sim_matrix(embedding_a, embedding_b)
            feature_similarity_matrix = feature_similarity_matrix.cpu()
            if self.training:
                feature_similarity_matrix = feature_similarity_matrix * mask
                #breakpoint()
            topk_similar_nodes_information = torch.topk(feature_similarity_matrix, self.k)
        fetch_similarity = torch.exp(topk_similar_nodes_information.values)
        fetch_similarity = fetch_similarity.to(self.device)
        fetch_label = []
        if self.criterion == 'sigmoid':
            one_hot_label = b_y
        elif self.criterion == 'softmax':
            one_hot_label = convert_to_one_hot_label(
                labels=b_y,
                num_classes=self.num_classes,
                batch_size=b_y.shape[0],
                device=self.device
            )
            one_hot_label = one_hot_label.to(self.device)

        for i in range(topk_similar_nodes_information.indices.shape[0]):
            fetch_label.append(
                torch.index_select(one_hot_label, 0, topk_similar_nodes_information.indices[i].to(self.device)))
        fetch_label = torch.stack(fetch_label)
        node_label_distribution = fetch_similarity.unsqueeze(-1) * fetch_label.to(self.device)
        total_fuse_distribution = node_label_distribution.sum(dim=1)

        return total_fuse_distribution


    def inference(self, x_all, subgraph_loader, training_embedding, training_y, stats_flag):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.layers))
        pbar.set_description(stats_flag)

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        xs = []
        p_sim_list = []
        for batch in subgraph_loader:
            batch = batch.to(self.device)
            x = x_all[batch.n_id.to(x_all.device)].to(self.device)
            x = self.layer_1(x, batch.edge_index)
            x = x.relu_()
            embedding = x[:batch.batch_size]
            y = batch.y[:batch.batch_size]
            p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                                  training_embedding,
                                                                  training_y,
                                                                  None))
            p_sim_list.append(p_sim)
            xs.append(x[:batch.batch_size].cpu())
            pbar.update(batch.batch_size)
        x_all = torch.cat(xs, dim=0)
        p_sim_total = torch.cat(p_sim_list, dim=0)

        xs = []
        for batch in subgraph_loader:
            batch = batch.to(self.device)
            x = x_all[batch.n_id.to(x_all.device)].to(self.device)
            x = self.layer_2(x, batch.edge_index)
            xs.append(x[:batch.batch_size].cpu())
            pbar.update(batch.batch_size)
        x_all = torch.cat(xs, dim=0)

        p_lc_total = self.m(x_all)
        final_label_distribution = self.eta * p_lc_total.to(self.device) + (1 - self.eta) * p_sim_total
        pbar.close()

        return final_label_distribution


class GAT_Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 criterion,
                 k,
                 eta,
                 device):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        self.layer_1 = GATConv(input_dim, out_channels=hidden_dim, heads=4)
        self.layer_2 = GATConv(4 * hidden_dim, hidden_dim, heads=4)
        self.layer_3 = GATConv(4 * hidden_dim, num_classes, heads=6, concat=False)
        self.layers.append(self.layer_1)
        self.layers.append(self.layer_2)
        self.layers.append(self.layer_3)

        self.criterion = criterion
        self.k = k
        self.eta = eta
        if criterion == 'sigmoid':
            self.m = nn.Sigmoid()
        elif criterion == 'softmax':
            self.m = nn.LogSoftmax(dim=1)
        self.num_classes = num_classes
        self.device = device


    def pretrain_with_batch(self, data):

        x, edge_index = data.x, data.edge_index
        x = F.elu(self.layer_1(x, edge_index))
        x = F.elu(self.layer_2(x, edge_index))
        lc_x = self.layer_3(x, edge_index)
        lc_x = lc_x[:data.batch_size]
        p_lc = self.m(lc_x)

        return p_lc


    @torch.no_grad()
    def get_embedding(self, data):
        dic = {}
        dic['full_data'] = data
        x, edge_index = data.x, data.edge_index
        #y = data.y[:data.batch_size]
        y = data.y[data.train_mask][:data.batch_size]
        x = F.elu(self.layer_1(x, edge_index))
        x = F.elu(self.layer_2(x, edge_index))
        #embedding = x[:data.batch_size]
        embedding = x[data.train_mask][:data.batch_size]

        dic['train_embedding'] = embedding
        dic['train_label'] = y

        return dic


    @torch.no_grad()
    def decode_z_q(self, train_info):

        data = train_info['full_data']
        x, edge_index = data.x, data.edge_index

        y = data.y[:data.batch_size]
        x = F.elu(self.layer_1(x, edge_index))
        x = F.elu(self.layer_2(x, edge_index))
        lc_x = self.layer_3(x, edge_index)
        #lc_x = lc_x[:data.batch_size]
        lc_x = lc_x[data.train_mask][:data.batch_size]
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(train_info['z_q'],
                                                              train_info['z_q'],
                                                              #y,
                                                              data.y[data.train_mask],
                                                              train_info['neighbors_info']))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution


    def forward(self,
                data,
                neighbors_info=None):
        x, edge_index = data.x, data.edge_index

        y = data.y[:data.batch_size]
        x = F.elu(self.layer_1(x, edge_index))
        x = F.elu(self.layer_2(x, edge_index))
        #embedding = x[:data.batch_size]
        embedding = x[data.train_mask][:data.batch_size]
        
        lc_x = self.layer_3(x, edge_index)
        #lc_x = lc_x[:data.batch_size]
        lc_x = lc_x[data.train_mask][:data.batch_size]
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                              embedding,
                                                              #y,
                                                              data.y[data.train_mask],
                                                              neighbors_info))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution, embedding


    def generate_label_with_similar_nodes(self,
                                          embedding_a,
                                          embedding_b,
                                          b_y,
                                          neighbors_info=None):
        if self.training:
            mask = torch.ones(b_y.shape[0], b_y.shape[0])
            torch.diagonal(mask, 0).zero_()
        with torch.no_grad():
            if torch.is_tensor(neighbors_info):
                # print("|> [generate_label_with_similar_nodes] Using neighbors information")
                feature_similarity_matrix = neighbors_info
            else:
                # print("|> [generate_label_with_similar_nodes] Not Using neighbors information")
                feature_similarity_matrix = sim_matrix(embedding_a, embedding_b)
            feature_similarity_matrix = feature_similarity_matrix.cpu()
            if self.training:
                feature_similarity_matrix = feature_similarity_matrix * mask
            topk_similar_nodes_information = torch.topk(feature_similarity_matrix, self.k)
        fetch_similarity = torch.exp(topk_similar_nodes_information.values)
        fetch_similarity = fetch_similarity.to(self.device)
        fetch_label = []
        if self.criterion == 'sigmoid':
            one_hot_label = b_y
        elif self.criterion == 'softmax':
            one_hot_label = convert_to_one_hot_label(
                labels=b_y,
                num_classes=self.num_classes,
                batch_size=b_y.shape[0],
                device=self.device
            )
            one_hot_label = one_hot_label.to(self.device)

        for i in range(topk_similar_nodes_information.indices.shape[0]):
            fetch_label.append(
                torch.index_select(one_hot_label, 0, topk_similar_nodes_information.indices[i].to(self.device)))
        fetch_label = torch.stack(fetch_label)
        node_label_distribution = fetch_similarity.unsqueeze(-1) * fetch_label.to(self.device)
        total_fuse_distribution = node_label_distribution.sum(dim=1)

        return total_fuse_distribution


    def inference(self, x_all, subgraph_loader, training_embedding, training_y, stats_flag):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.layers))
        pbar.set_description(stats_flag)

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        xs = []
        p_sim_list = []
        for batch in subgraph_loader:
            batch = batch.to(self.device)
            x = x_all[batch.n_id.to(x_all.device)].to(self.device)
            x = F.elu(self.layer_1(x, batch.edge_index))
            xs.append(x[:batch.batch_size].cpu())
            pbar.update(batch.batch_size)
        x_all = torch.cat(xs, dim=0)

        xs = []
        for batch in subgraph_loader:
            batch = batch.to(self.device)
            x = x_all[batch.n_id.to(x_all.device)].to(self.device)
            x = F.elu(self.layer_2(x, batch.edge_index))
            embedding = x[:batch.batch_size]
            p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                                  training_embedding,
                                                                  training_y,
                                                                  None))
            p_sim_list.append(p_sim)
            xs.append(x[:batch.batch_size].cpu())
            pbar.update(batch.batch_size)
        x_all = torch.cat(xs, dim=0)
        p_sim_total = torch.cat(p_sim_list, dim=0)

        xs = []
        for batch in subgraph_loader:
            batch = batch.to(self.device)
            x = x_all[batch.n_id.to(x_all.device)].to(self.device)
            x = self.layer_3(x, batch.edge_index)
            xs.append(x[:batch.batch_size].cpu())
            pbar.update(batch.batch_size)
        x_all = torch.cat(xs, dim=0)

        p_lc_total = self.m(x_all)
        final_label_distribution = self.eta * p_lc_total.to(self.device) + (1 - self.eta) * p_sim_total
        pbar.close()

        return final_label_distribution


class GATv2Conv_Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 criterion,
                 k,
                 eta,
                 device):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        self.layer_1 = GATv2Conv(input_dim, out_channels=hidden_dim, heads=4)
        self.layer_2 = GATv2Conv(4 * hidden_dim, hidden_dim, heads=4)
        self.layer_3 = GATv2Conv(4 * hidden_dim, num_classes, heads=6, concat=False)
        self.layers.append(self.layer_1)
        self.layers.append(self.layer_2)
        self.layers.append(self.layer_3)

        self.criterion = criterion
        self.k = k
        self.eta = eta
        if criterion == 'sigmoid':
            self.m = nn.Sigmoid()
        elif criterion == 'softmax':
            self.m = nn.LogSoftmax(dim=1)
        self.num_classes = num_classes
        self.device = device


    def pretrain_with_batch(self, data):

        x, edge_index = data.x, data.edge_index
        x = F.elu(self.layer_1(x, edge_index))
        x = F.elu(self.layer_2(x, edge_index))
        lc_x = self.layer_3(x, edge_index)
        lc_x = lc_x[:data.batch_size]
        p_lc = self.m(lc_x)

        return p_lc


    @torch.no_grad()
    def get_embedding(self, data):
        dic = {}
        dic['full_data'] = data
        x, edge_index = data.x, data.edge_index
        #y = data.y[:data.batch_size]
        y = data.y[data.train_mask][:data.batch_size]
        x = F.elu(self.layer_1(x, edge_index))
        x = F.elu(self.layer_2(x, edge_index))
        #embedding = x[:data.batch_size]
        embedding = x[data.train_mask][:data.batch_size]

        dic['train_embedding'] = embedding
        dic['train_label'] = y

        return dic


    @torch.no_grad()
    def decode_z_q(self, train_info):

        data = train_info['full_data']
        x, edge_index = data.x, data.edge_index

        y = data.y[:data.batch_size]
        x = F.elu(self.layer_1(x, edge_index))
        x = F.elu(self.layer_2(x, edge_index))
        lc_x = self.layer_3(x, edge_index)
        #lc_x = lc_x[:data.batch_size]
        lc_x = lc_x[data.train_mask][:data.batch_size]
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(train_info['z_q'],
                                                              train_info['z_q'],
                                                              #y,
                                                              data.y[data.train_mask],
                                                              train_info['neighbors_info']))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution


    def forward(self,
                data,
                neighbors_info=None):
        x, edge_index = data.x, data.edge_index

        y = data.y[:data.batch_size]
        x = F.elu(self.layer_1(x, edge_index))
        x = F.elu(self.layer_2(x, edge_index))
        #embedding = x[:data.batch_size]
        embedding = x[data.train_mask][:data.batch_size]

        lc_x = self.layer_3(x, edge_index)
        #lc_x = lc_x[:data.batch_size]
        lc_x = lc_x[data.train_mask][:data.batch_size]
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                              embedding,
                                                              #y,
                                                              data.y[data.train_mask],
                                                              neighbors_info))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution, embedding


    def generate_label_with_similar_nodes(self,
                                          embedding_a,
                                          embedding_b,
                                          b_y,
                                          neighbors_info=None):
        if self.training:
            mask = torch.ones(b_y.shape[0], b_y.shape[0])
            torch.diagonal(mask, 0).zero_()
        with torch.no_grad():
            if torch.is_tensor(neighbors_info):
                # print("|> [generate_label_with_similar_nodes] Using neighbors information")
                feature_similarity_matrix = neighbors_info
            else:
                # print("|> [generate_label_with_similar_nodes] Not Using neighbors information")
                feature_similarity_matrix = sim_matrix(embedding_a, embedding_b)
            feature_similarity_matrix = feature_similarity_matrix.cpu()
            if self.training:
                feature_similarity_matrix = feature_similarity_matrix * mask
            topk_similar_nodes_information = torch.topk(feature_similarity_matrix, self.k)
        fetch_similarity = torch.exp(topk_similar_nodes_information.values)
        fetch_similarity = fetch_similarity.to(self.device)
        fetch_label = []
        if self.criterion == 'sigmoid':
            one_hot_label = b_y
        elif self.criterion == 'softmax':
            one_hot_label = convert_to_one_hot_label(
                labels=b_y,
                num_classes=self.num_classes,
                batch_size=b_y.shape[0],
                device=self.device
            )
            one_hot_label = one_hot_label.to(self.device)

        for i in range(topk_similar_nodes_information.indices.shape[0]):
            fetch_label.append(
                torch.index_select(one_hot_label, 0, topk_similar_nodes_information.indices[i].to(self.device)))
        fetch_label = torch.stack(fetch_label)
        node_label_distribution = fetch_similarity.unsqueeze(-1) * fetch_label.to(self.device)
        total_fuse_distribution = node_label_distribution.sum(dim=1)

        return total_fuse_distribution


    def inference(self, x_all, subgraph_loader, training_embedding, training_y, stats_flag):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.layers))
        pbar.set_description(stats_flag)

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        xs = []
        p_sim_list = []
        for batch in subgraph_loader:
            batch = batch.to(self.device)
            x = x_all[batch.n_id.to(x_all.device)].to(self.device)
            x = F.elu(self.layer_1(x, batch.edge_index))
            xs.append(x[:batch.batch_size].cpu())
            pbar.update(batch.batch_size)
        x_all = torch.cat(xs, dim=0)

        xs = []
        for batch in subgraph_loader:
            batch = batch.to(self.device)
            x = x_all[batch.n_id.to(x_all.device)].to(self.device)
            x = F.elu(self.layer_2(x, batch.edge_index))
            embedding = x[:batch.batch_size]
            p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                                  training_embedding,
                                                                  training_y,
                                                                  None))
            p_sim_list.append(p_sim)
            xs.append(x[:batch.batch_size].cpu())
            pbar.update(batch.batch_size)
        x_all = torch.cat(xs, dim=0)
        p_sim_total = torch.cat(p_sim_list, dim=0)

        xs = []
        for batch in subgraph_loader:
            batch = batch.to(self.device)
            x = x_all[batch.n_id.to(x_all.device)].to(self.device)
            x = self.layer_3(x, batch.edge_index)
            xs.append(x[:batch.batch_size].cpu())
            pbar.update(batch.batch_size)
        x_all = torch.cat(xs, dim=0)

        p_lc_total = self.m(x_all)
        final_label_distribution = self.eta * p_lc_total.to(self.device) + (1 - self.eta) * p_sim_total
        pbar.close()

        return final_label_distribution


class GCN_Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 criterion,
                 k,
                 eta,
                 device):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        self.layer_1 = GCNConv(input_dim, hidden_dim)
        self.layer_2 = GCNConv(hidden_dim, num_classes)
        self.layers.append(self.layer_1)
        self.layers.append(self.layer_2)

        self.criterion = criterion
        self.k = k
        self.eta = eta
        if criterion == 'sigmoid':
            self.m = nn.Sigmoid()
        elif criterion == 'softmax':
            self.m = nn.LogSoftmax(dim=1)
        self.num_classes = num_classes
        self.device = device

    def pretrain_with_batch(self, data):

        x, edge_index = data.x, data.edge_index

        x = self.layer_1(x, edge_index)
        x = F.relu(x)
        lc_x = self.layer_2(x, edge_index)

        lc_x = lc_x[:data.batch_size]
        p_lc = self.m(lc_x)

        return p_lc

    @torch.no_grad()
    def get_embedding(self, data):
        dic = {}
        dic['full_data'] = data
        x, edge_index = data.x, data.edge_index
        #y = data.y[:data.batch_size]
        y = data.y[data.train_mask][:data.batch_size]
        x = self.layer_1(x, edge_index)
        x = F.relu(x)
        #embedding = x[:data.batch_size]
        embedding = x[data.train_mask][:data.batch_size]

        dic['train_embedding'] = embedding
        dic['train_label'] = y

        return dic

    @torch.no_grad()
    def decode_z_q(self, train_info):

        data = train_info['full_data']
        x, edge_index = data.x, data.edge_index

        y = data.y[:data.batch_size]
        x = self.layer_1(x, edge_index)
        x = F.relu(x)
        lc_x = self.layer_2(x, edge_index)
        #lc_x = lc_x[:data.batch_size]
        lc_x = lc_x[data.train_mask][:data.batch_size]
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(train_info['z_q'],
                                                              train_info['z_q'],
                                                              #y,
                                                              data.y[data.train_mask],
                                                              train_info['neighbors_info']))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution

    def forward(self,
                data,
                neighbors_info=None):
        x, edge_index = data.x, data.edge_index

        y = data.y[:data.batch_size]
        x = self.layer_1(x, edge_index)
        x = F.relu(x)
        #embedding = x[:data.batch_size]
        embedding = x[data.train_mask][:data.batch_size]

        lc_x = self.layer_2(x, edge_index)
        #lc_x = lc_x[:data.batch_size]
        lc_x = lc_x[data.train_mask][:data.batch_size]
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                              embedding,
                                                              #y,
                                                              data.y[data.train_mask],
                                                              neighbors_info))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution, embedding

    def generate_label_with_similar_nodes(self,
                                          embedding_a,
                                          embedding_b,
                                          b_y,
                                          neighbors_info=None):
        if self.training:
            mask = torch.ones(b_y.shape[0], b_y.shape[0])
            torch.diagonal(mask, 0).zero_()
        with torch.no_grad():
            if torch.is_tensor(neighbors_info):
                # print("|> [generate_label_with_similar_nodes] Using neighbors information")
                feature_similarity_matrix = neighbors_info
            else:
                # print("|> [generate_label_with_similar_nodes] Not Using neighbors information")
                feature_similarity_matrix = sim_matrix(embedding_a, embedding_b)
            feature_similarity_matrix = feature_similarity_matrix.cpu()
            if self.training:
                feature_similarity_matrix = feature_similarity_matrix * mask
            topk_similar_nodes_information = torch.topk(feature_similarity_matrix, self.k)
        fetch_similarity = torch.exp(topk_similar_nodes_information.values)
        fetch_similarity = fetch_similarity.to(self.device)
        fetch_label = []
        if self.criterion == 'sigmoid':
            one_hot_label = b_y
        elif self.criterion == 'softmax':
            one_hot_label = convert_to_one_hot_label(
                labels=b_y,
                num_classes=self.num_classes,
                batch_size=b_y.shape[0],
                device=self.device
            )
            one_hot_label = one_hot_label.to(self.device)

        for i in range(topk_similar_nodes_information.indices.shape[0]):
            fetch_label.append(
                torch.index_select(one_hot_label, 0, topk_similar_nodes_information.indices[i].to(self.device)))
        fetch_label = torch.stack(fetch_label)
        node_label_distribution = fetch_similarity.unsqueeze(-1) * fetch_label.to(self.device)
        total_fuse_distribution = node_label_distribution.sum(dim=1)

        return total_fuse_distribution

    def inference(self, x_all, subgraph_loader, training_embedding, training_y, stats_flag):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.layers))
        pbar.set_description(stats_flag)

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        xs = []
        p_sim_list = []
        for batch in subgraph_loader:
            batch = batch.to(self.device)
            x = x_all[batch.n_id.to(x_all.device)].to(self.device)
            x = self.layer_1(x, batch.edge_index)
            x = F.relu(x)
            embedding = x[:batch.batch_size]
            p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                                  training_embedding,
                                                                  training_y,
                                                                  None))
            p_sim_list.append(p_sim)
            xs.append(x[:batch.batch_size].cpu())
            pbar.update(batch.batch_size)
        x_all = torch.cat(xs, dim=0)
        p_sim_total = torch.cat(p_sim_list, dim=0)

        xs = []
        for batch in subgraph_loader:
            batch = batch.to(self.device)
            x = x_all[batch.n_id.to(x_all.device)].to(self.device)
            x = self.layer_2(x, batch.edge_index)
            xs.append(x[:batch.batch_size].cpu())
            pbar.update(batch.batch_size)
        x_all = torch.cat(xs, dim=0)

        p_lc_total = self.m(x_all)
        final_label_distribution = self.eta * p_lc_total.to(self.device) + (1 - self.eta) * p_sim_total
        pbar.close()

        return final_label_distribution


class SGC_Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 criterion,
                 k,
                 eta,
                 device):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        self.layer_1 = SGConv(input_dim, hidden_dim, K=2)
        self.layer_2 = SGConv(hidden_dim, num_classes, K=2)
        self.layers.append(self.layer_1)
        self.layers.append(self.layer_2)

        self.criterion = criterion
        self.k = k
        self.eta = eta
        if criterion == 'sigmoid':
            self.m = nn.Sigmoid()
        elif criterion == 'softmax':
            self.m = nn.LogSoftmax(dim=1)
        self.num_classes = num_classes
        self.device = device

    def pretrain_with_batch(self, data):

        x, edge_index = data.x, data.edge_index

        x = self.layer_1(x, edge_index)
        lc_x = self.layer_2(x, edge_index)

        lc_x = lc_x[:data.batch_size]
        p_lc = self.m(lc_x)

        return p_lc

    @torch.no_grad()
    def get_embedding(self, data):
        dic = {}
        dic['full_data'] = data
        x, edge_index = data.x, data.edge_index
        #y = data.y[:data.batch_size]
        y = data.y[data.train_mask][:data.batch_size]
        x = self.layer_1(x, edge_index)
        #embedding = x[:data.batch_size]
        embedding = x[data.train_mask][:data.batch_size]

        dic['train_embedding'] = embedding
        dic['train_label'] = y

        return dic

    @torch.no_grad()
    def decode_z_q(self, train_info):

        data = train_info['full_data']
        x, edge_index = data.x, data.edge_index

        y = data.y[:data.batch_size]
        x = self.layer_1(x, edge_index)
        lc_x = self.layer_2(x, edge_index)
        #lc_x = lc_x[:data.batch_size]
        lc_x = lc_x[data.train_mask][:data.batch_size]
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(train_info['z_q'],
                                                              train_info['z_q'],
                                                              #y,
                                                              data.y[data.train_mask],
                                                              train_info['neighbors_info']))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution

    def forward(self,
                data,
                neighbors_info=None):
        x, edge_index = data.x, data.edge_index

        y = data.y[:data.batch_size]
        x = self.layer_1(x, edge_index)
        #embedding = x[:data.batch_size]
        embedding = x[data.train_mask][:data.batch_size]

        lc_x = self.layer_2(x, edge_index)
        #lc_x = lc_x[:data.batch_size]
        lc_x = lc_x[data.train_mask][:data.batch_size]
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                              embedding,
                                                              #y,
                                                              data.y[data.train_mask],
                                                              neighbors_info))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution, embedding

    def generate_label_with_similar_nodes(self,
                                          embedding_a,
                                          embedding_b,
                                          b_y,
                                          neighbors_info=None):
        if self.training:
            mask = torch.ones(b_y.shape[0], b_y.shape[0])
            torch.diagonal(mask, 0).zero_()
        with torch.no_grad():
            if torch.is_tensor(neighbors_info):
                # print("|> [generate_label_with_similar_nodes] Using neighbors information")
                feature_similarity_matrix = neighbors_info
            else:
                # print("|> [generate_label_with_similar_nodes] Not Using neighbors information")
                feature_similarity_matrix = sim_matrix(embedding_a, embedding_b)
            feature_similarity_matrix = feature_similarity_matrix.cpu()
            if self.training:
                feature_similarity_matrix = feature_similarity_matrix * mask
            topk_similar_nodes_information = torch.topk(feature_similarity_matrix, self.k)
        fetch_similarity = torch.exp(topk_similar_nodes_information.values)
        fetch_similarity = fetch_similarity.to(self.device)
        fetch_label = []
        if self.criterion == 'sigmoid':
            one_hot_label = b_y
        elif self.criterion == 'softmax':
            one_hot_label = convert_to_one_hot_label(
                labels=b_y,
                num_classes=self.num_classes,
                batch_size=b_y.shape[0],
                device=self.device
            )
            one_hot_label = one_hot_label.to(self.device)

        for i in range(topk_similar_nodes_information.indices.shape[0]):
            fetch_label.append(
                torch.index_select(one_hot_label, 0, topk_similar_nodes_information.indices[i].to(self.device)))
        fetch_label = torch.stack(fetch_label)
        node_label_distribution = fetch_similarity.unsqueeze(-1) * fetch_label.to(self.device)
        total_fuse_distribution = node_label_distribution.sum(dim=1)

        return total_fuse_distribution

    def inference(self, x_all, subgraph_loader, training_embedding, training_y, stats_flag):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.layers))
        pbar.set_description(stats_flag)

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        xs = []
        p_sim_list = []
        for batch in subgraph_loader:
            batch = batch.to(self.device)
            x = x_all[batch.n_id.to(x_all.device)].to(self.device)
            x = self.layer_1(x, batch.edge_index)
            embedding = x[:batch.batch_size]
            p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                                  training_embedding,
                                                                  training_y,
                                                                  None))
            p_sim_list.append(p_sim)
            xs.append(x[:batch.batch_size].cpu())
            pbar.update(batch.batch_size)

        x_all = torch.cat(xs, dim=0)
        p_sim_total = torch.cat(p_sim_list, dim=0)

        xs = []
        for batch in subgraph_loader:
            batch = batch.to(self.device)
            x = x_all[batch.n_id.to(x_all.device)].to(self.device)
            x = self.layer_2(x, batch.edge_index)
            xs.append(x[:batch.batch_size].cpu())
            pbar.update(batch.batch_size)
        x_all = torch.cat(xs, dim=0)

        p_lc_total = self.m(x_all)
        final_label_distribution = self.eta * p_lc_total.to(self.device) + (1 - self.eta) * p_sim_total
        pbar.close()

        return final_label_distribution


class GCNII_Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 criterion,
                 k,
                 eta,
                 device,
                 alpha=0.5,
                 theta=1.0,
                 n_layers=9,
                 shared_weights=False,
                 dropout=0.2):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(input_dim, hidden_dim))
        self.lins.append(torch.nn.Linear(hidden_dim, num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(n_layers):
            self.convs.append(
                GCN2Conv(hidden_dim, alpha, theta,
                         layer + 1, shared_weights, normalize=False))
        self.dropout = dropout

        self.criterion = criterion
        self.k = k
        self.eta = eta
        if criterion == 'sigmoid':
            self.m = nn.Sigmoid()
        elif criterion == 'softmax':
            self.m = nn.LogSoftmax(dim=1)
        self.num_classes = num_classes
        self.device = device

    def pretrain_with_batch(self, data):

        x, adj_t = data.x, data.adj_t
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for i, conv in enumerate(self.convs):
            h = F.dropout(x, self.dropout, training=self.training)
            h = conv(h, x_0, adj_t)
            x = x + h
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        lc_x = self.lins[1](x)
        lc_x = lc_x[:data.batch_size]
        p_lc = self.m(lc_x)

        return p_lc

    @torch.no_grad()
    def get_embedding(self, data):
        dic = {}
        dic['full_data'] = data

        x, adj_t = data.x, data.adj_t
        x = F.dropout(x, self.dropout, training=False)
        x = x_0 = self.lins[0](x).relu()
        #y = data.y[:data.batch_size]
        y = data.y[data.train_mask][:data.batch_size]
        for i, conv in enumerate(self.convs):
            h = F.dropout(x, self.dropout, training=False)
            h = conv(h, x_0, adj_t)
            x = x + h
            x = x.relu()

        #embedding = x[:data.batch_size]
        embedding = x[data.train_mask][:data.batch_size]

        dic['train_embedding'] = embedding
        dic['train_label'] = y

        return dic

    @torch.no_grad()
    def decode_z_q(self, train_info):

        data = train_info['full_data']

        x, adj_t = data.x, data.adj_t
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        y = data.y[:data.batch_size]

        for i, conv in enumerate(self.convs):
            h = F.dropout(x, self.dropout, training=self.training)
            h = conv(h, x_0, adj_t)
            x = x + h
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        lc_x = self.lins[1](x)
        #lc_x = lc_x[:data.batch_size]
        lc_x = lc_x[data.train_mask][:data.batch_size]
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(train_info['z_q'],
                                                              train_info['z_q'],
                                                              #y,
                                                              data.y[data.train_mask],
                                                              train_info['neighbors_info']))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution

    def forward(self,
                data,
                neighbors_info=None):
        x, adj_t = data.x, data.adj_t
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        y = data.y[:data.batch_size]

        for i, conv in enumerate(self.convs):
            h = F.dropout(x, self.dropout, training=self.training)
            h = conv(h, x_0, adj_t)
            x = x + h
            x = x.relu()

        #embedding = x[:data.batch_size]
        embedding = x[data.train_mask][:data.batch_size]

        x = F.dropout(x, self.dropout, training=self.training)
        lc_x = self.lins[1](x)
        #lc_x = lc_x[:data.batch_size]
        lc_x = lc_x[data.train_mask][:data.batch_size]
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                              embedding,
                                                              #y,
                                                              data.y[data.train_mask],
                                                              neighbors_info))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution, embedding

    def generate_label_with_similar_nodes(self,
                                          embedding_a,
                                          embedding_b,
                                          b_y,
                                          neighbors_info=None):
        if self.training:
            mask = torch.ones(b_y.shape[0], b_y.shape[0])
            torch.diagonal(mask, 0).zero_()
        with torch.no_grad():
            if torch.is_tensor(neighbors_info):
                # print("|> [generate_label_with_similar_nodes] Using neighbors information")
                feature_similarity_matrix = neighbors_info
            else:
                # print("|> [generate_label_with_similar_nodes] Not Using neighbors information")
                feature_similarity_matrix = sim_matrix(embedding_a, embedding_b)
            feature_similarity_matrix = feature_similarity_matrix.cpu()
            if self.training:
                feature_similarity_matrix = feature_similarity_matrix * mask
            topk_similar_nodes_information = torch.topk(feature_similarity_matrix, self.k)
        fetch_similarity = torch.exp(topk_similar_nodes_information.values)
        fetch_similarity = fetch_similarity.to(self.device)
        fetch_label = []
        if self.criterion == 'sigmoid':
            one_hot_label = b_y
        elif self.criterion == 'softmax':
            one_hot_label = convert_to_one_hot_label(
                labels=b_y,
                num_classes=self.num_classes,
                batch_size=b_y.shape[0],
                device=self.device
            )
            one_hot_label = one_hot_label.to(self.device)

        for i in range(topk_similar_nodes_information.indices.shape[0]):
            fetch_label.append(
                torch.index_select(one_hot_label, 0, topk_similar_nodes_information.indices[i].to(self.device)))
        fetch_label = torch.stack(fetch_label)
        node_label_distribution = fetch_similarity.unsqueeze(-1) * fetch_label.to(self.device)
        total_fuse_distribution = node_label_distribution.sum(dim=1)

        return total_fuse_distribution

    def inference(self, x_all, subgraph_loader, training_embedding, training_y, stats_flag):
        pbar = tqdm(total=len(subgraph_loader.dataset) * (len(self.convs) + len(self.lins)))
        pbar.set_description(stats_flag)

        xs = []
        for batch in subgraph_loader:
            x = x_all[batch.n_id.to(x_all.device)].to(self.device)
            x = self.lins[0](x).relu()
            xs.append(x[:batch.batch_size].cpu())
            pbar.update(batch.batch_size)
        x_all = x_all_first = torch.cat(xs, dim=0)
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:

        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(self.device)
                x_0 = x_all_first[batch.n_id.to(x_all.device)].to(self.device)
                h = F.dropout(x, self.dropout, training=self.training)
                h = conv(h, x_0, batch.adj_t.to(self.device))
                x = h + x
                x = x.relu()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)

        embedding_all = copy.copy(x_all)

        xs = []
        p_sim_list = []
        for batch in subgraph_loader:
            x = x_all[batch.n_id.to(x_all.device)].to(self.device)
            embedding = embedding_all[batch.n_id.to(x_all.device)].to(self.device)[:batch.batch_size]
            # print(embedding[:])
            p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                                  training_embedding,
                                                                  training_y,
                                                                  None))
            p_sim_list.append(p_sim)
            x = self.lins[1](x)
            xs.append(x[:batch.batch_size].cpu())
            pbar.update(batch.batch_size)
        x_all = torch.cat(xs, dim=0)
        p_sim_total = torch.cat(p_sim_list, dim=0)

        p_lc_total = self.m(x_all)
        final_label_distribution = self.eta * p_lc_total.to(self.device) + (1 - self.eta) * p_sim_total
        pbar.close()

        return final_label_distribution


class FILM_Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes,
                 criterion,
                 k,
                 eta,
                 device,
                 hidden_dim=320,
                 dropout=0.1,
                 n_layers=4,):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(FiLMConv(input_dim, hidden_dim))
        for _ in range(n_layers - 2):
            self.convs.append(FiLMConv(hidden_dim, hidden_dim))
        self.convs.append(FiLMConv(hidden_dim, num_classes, act=None))

        self.norms = torch.nn.ModuleList()
        for _ in range(n_layers - 1):
            self.norms.append(BatchNorm1d(hidden_dim))

        self.criterion = criterion
        self.k = k
        self.eta = eta
        if criterion == 'sigmoid':
            self.m = nn.Sigmoid()
        elif criterion == 'softmax':
            self.m = nn.LogSoftmax(dim=1)

        self.num_classes = num_classes
        self.device = device
        self.dropout = dropout

    def pretrain_with_batch(self, data):
        """
        Pretrain
        :param data:
        :return:
        """
        x, edge_index = data.x, data.edge_index

        for conv, norm in zip(self.convs[:-1], self.norms):
            x = norm(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        lc_x = self.convs[-1](x, edge_index)
        lc_x = lc_x[:data.batch_size]
        p_lc = self.m(lc_x)
        return p_lc

    @torch.no_grad()
    def get_embedding(self, data):
        dic = {}
        dic['full_data'] = data

        x, edge_index = data.x, data.edge_index
        #y = data.y[:data.batch_size]
        y = data.y[data.train_mask][:data.batch_size]
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = norm(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=False)

        #embedding = x[:data.batch_size]
        embedding = x[data.train_mask][:data.batch_size]

        dic['train_embedding'] = embedding
        dic['train_label'] = y

        return dic

    @torch.no_grad()
    def decode_z_q(self, train_info):

        data = train_info['full_data']

        x, edge_index = data.x, data.edge_index

        #y = data.y[:data.batch_size]
        y = data.y[data.train_mask][:data.batch_size]
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = norm(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=False)

        lc_x = self.convs[-1](x, edge_index)
        #lc_x = lc_x[:data.batch_size]
        lc_x = lc_x[data.train_mask][:data.batch_size]
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(train_info['z_q'],
                                                              train_info['z_q'],
                                                              #y,
                                                              data.y[data.train_mask],
                                                              train_info['neighbors_info']))

        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution

    def forward(self,
                data,
                neighbors_info=None):

        x, edge_index = data.x, data.edge_index

        y = data.y[:data.batch_size]

        for conv, norm in zip(self.convs[:-1], self.norms):
            x = norm(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=False)

        #embedding = x[:data.batch_size]
        embedding = x[data.train_mask][:data.batch_size]

        lc_x = self.convs[-1](x, edge_index)
        #lc_x = lc_x[:data.batch_size]
        lc_x = lc_x[data.train_mask][:data.batch_size]
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                              embedding,
                                                              #y,
                                                              data.y[data.train_mask],
                                                              neighbors_info))

        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution, embedding

    def generate_label_with_similar_nodes(self,
                                          embedding_a,
                                          embedding_b,
                                          b_y,
                                          neighbors_info=None):
        if self.training:
            mask = torch.ones(b_y.shape[0], b_y.shape[0])
            torch.diagonal(mask, 0).zero_()
        with torch.no_grad():
            if torch.is_tensor(neighbors_info):
                # print("|> [generate_label_with_similar_nodes] Using neighbors information")
                feature_similarity_matrix = neighbors_info
            else:
                # print("|> [generate_label_with_similar_nodes] Not Using neighbors information")
                feature_similarity_matrix = sim_matrix(embedding_a, embedding_b)
            feature_similarity_matrix = feature_similarity_matrix.cpu()
            if self.training:
                feature_similarity_matrix = feature_similarity_matrix * mask
            topk_similar_nodes_information = torch.topk(
                feature_similarity_matrix, self.k)
        fetch_similarity = torch.exp(topk_similar_nodes_information.values)
        fetch_similarity = fetch_similarity.to(self.device)
        fetch_label = []
        if self.criterion == 'sigmoid':
            one_hot_label = b_y
        elif self.criterion == 'softmax':
            one_hot_label = convert_to_one_hot_label(
                labels=b_y,
                num_classes=self.num_classes,
                batch_size=b_y.shape[0],
                device=self.device
            )
            one_hot_label = one_hot_label.to(self.device)

        for i in range(topk_similar_nodes_information.indices.shape[0]):
            fetch_label.append(
                torch.index_select(one_hot_label, 0,
                                   topk_similar_nodes_information.indices[i].to(
                                       self.device)))
        fetch_label = torch.stack(fetch_label)
        node_label_distribution = fetch_similarity.unsqueeze(
            -1) * fetch_label.to(self.device)
        total_fuse_distribution = node_label_distribution.sum(dim=1)

        return total_fuse_distribution

    def inference(self, x_all,
                  subgraph_loader,
                  training_embedding,
                  training_y,
                  stats_flag):
        pbar = tqdm(total=len(subgraph_loader.dataset) * (
                len(self.convs)))
        pbar.set_description(stats_flag)

        # Get the embedding first
        for conv, norm in zip(self.convs[:-1], self.norms):
            xs = []
            for batch in subgraph_loader:
                batch = batch.to(self.device)
                x = x_all[batch.n_id.to(x_all.device)].to(self.device)
                x = norm(conv(x, batch.edge_index))
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)

        embedding_all = copy.copy(x_all)

        xs = []
        p_sim_list = []
        for batch in subgraph_loader:
            batch = batch.to(self.device)
            x = x_all[batch.n_id.to(x_all.device)].to(self.device)
            embedding = embedding_all[batch.n_id.to(x_all.device)].to(
                self.device)[:batch.batch_size]
            p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                                  training_embedding,
                                                                  training_y,
                                                                  None))
            p_sim_list.append(p_sim)
            x = self.convs[-1](x, batch.edge_index)
            xs.append(x[:batch.batch_size].cpu())
            pbar.update(batch.batch_size)
        x_all = torch.cat(xs, dim=0)
        p_sim_total = torch.cat(p_sim_list, dim=0)

        p_lc_total = self.m(x_all)
        final_label_distribution = self.eta * p_lc_total.to(self.device) + \
                                   (1 - self.eta) * p_sim_total
        pbar.close()

        return final_label_distribution


class DNA_Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes,
                 criterion,
                 k,
                 eta,
                 device,
                 hidden_dim=128,
                 n_layers=4,
                 heads=8,
                 groups=8
                 ):
        super().__init__()

        self.hidden_channels = hidden_dim
        self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
        self.convs = torch.nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(
                DNAConv(hidden_dim, heads, groups, dropout=0.8,
                        cached=False)
            )
        self.lin2 = torch.nn.Linear(hidden_dim, num_classes)

        self.criterion = criterion
        self.k = k
        self.eta = eta
        if criterion == 'sigmoid':
            self.m = nn.Sigmoid()
        elif criterion == 'softmax':
            self.m = nn.LogSoftmax(dim=1)

        self.num_classes = num_classes
        self.device = device

    def pretrain_with_batch(self, data):
        """
        :param data:
        :return:
        """
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x_all = x.view(-1, 1, self.hidden_channels)
        for conv in self.convs:
            x = F.relu(conv(x_all, edge_index))
            x = x.view(-1, 1, self.hidden_channels)
            x_all = torch.cat([x_all, x], dim=1)
        x = x_all[:, -1]
        x = F.dropout(x, p=0.5, training=self.training)
        lc_x = self.lin2(x)
        lc_x = lc_x[:data.batch_size]

        p_lc = self.m(lc_x)
        return p_lc

    @torch.no_grad()
    def get_embedding(self, data):
        dic = {}
        dic['full_data'] = data

        x, edge_index = data.x, data.edge_index
        #y = data.y[:data.batch_size]
        y = data.y[data.train_mask][:data.batch_size]
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x_all = x.view(-1, 1, self.hidden_channels)
        for conv in self.convs:
            x = F.relu(conv(x_all, edge_index))
            x = x.view(-1, 1, self.hidden_channels)
            x_all = torch.cat([x_all, x], dim=1)
        x = x_all[:, -1]
        x = F.dropout(x, p=0.5, training=False)

        #embedding = x[:data.batch_size]
        embedding = x[data.train_mask][:data.batch_size]

        dic['train_embedding'] = embedding
        dic['train_label'] = y

        return dic

    @torch.no_grad()
    def decode_z_q(self, train_info):

        data = train_info['full_data']

        x, edge_index = data.x, data.edge_index

        y = data.y[:data.batch_size]

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x_all = x.view(-1, 1, self.hidden_channels)
        for conv in self.convs:
            x = F.relu(conv(x_all, edge_index))
            x = x.view(-1, 1, self.hidden_channels)
            x_all = torch.cat([x_all, x], dim=1)
        x = x_all[:, -1]
        x = F.dropout(x, p=0.5, training=self.training)
        lc_x = self.lin2(x)
        #lc_x = lc_x[:data.batch_size]
        lc_x = lc_x[data.train_mask][:data.batch_size]

        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(train_info['z_q'],
                                                              train_info['z_q'],
                                                              #y,
                                                              data.y[data.train_mask],
                                                              train_info[
                                                                  'neighbors_info']))

        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution

    def forward(self,
                data,
                neighbors_info=None):

        x, edge_index = data.x, data.edge_index

        y = data.y[:data.batch_size]

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x_all = x.view(-1, 1, self.hidden_channels)
        for conv in self.convs:
            x = F.relu(conv(x_all, edge_index))
            x = x.view(-1, 1, self.hidden_channels)
            x_all = torch.cat([x_all, x], dim=1)
        x = x_all[:, -1]
        x = F.dropout(x, p=0.5, training=self.training)

        #embedding = x[:data.batch_size]
        embedding = x[data.train_mask][:data.batch_size]

        lc_x = self.lin2(x)
        #lc_x = lc_x[:data.batch_size]
        lc_x = lc_x[data.train_mask][:data.batch_size]
        p_lc = self.m(lc_x)
        p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                              embedding,
                                                              #y,
                                                              data.y[data.train_mask],
                                                              neighbors_info))

        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution, embedding

    def generate_label_with_similar_nodes(self,
                                          embedding_a,
                                          embedding_b,
                                          b_y,
                                          neighbors_info=None):
        if self.training:
            mask = torch.ones(b_y.shape[0], b_y.shape[0])
            torch.diagonal(mask, 0).zero_()
        with torch.no_grad():
            if torch.is_tensor(neighbors_info):
                # print("|> [generate_label_with_similar_nodes] Using neighbors information")
                feature_similarity_matrix = neighbors_info
            else:
                # print("|> [generate_label_with_similar_nodes] Not Using neighbors information")
                feature_similarity_matrix = sim_matrix(embedding_a, embedding_b)
            feature_similarity_matrix = feature_similarity_matrix.cpu()
            if self.training:
                feature_similarity_matrix = feature_similarity_matrix * mask
            topk_similar_nodes_information = torch.topk(
                feature_similarity_matrix, self.k)
        fetch_similarity = torch.exp(topk_similar_nodes_information.values)
        fetch_similarity = fetch_similarity.to(self.device)
        fetch_label = []
        if self.criterion == 'sigmoid':
            one_hot_label = b_y
        elif self.criterion == 'softmax':
            one_hot_label = convert_to_one_hot_label(
                labels=b_y,
                num_classes=self.num_classes,
                batch_size=b_y.shape[0],
                device=self.device
            )
            one_hot_label = one_hot_label.to(self.device)

        for i in range(topk_similar_nodes_information.indices.shape[0]):
            fetch_label.append(
                torch.index_select(one_hot_label, 0,
                                   topk_similar_nodes_information.indices[i].to(
                                       self.device)))
        fetch_label = torch.stack(fetch_label)
        node_label_distribution = fetch_similarity.unsqueeze(
            -1) * fetch_label.to(self.device)
        total_fuse_distribution = node_label_distribution.sum(dim=1)

        return total_fuse_distribution

    def inference(self, x_all,
                  subgraph_loader,
                  training_embedding,
                  training_y,
                  stats_flag):

        pbar = tqdm(total=len(subgraph_loader.dataset) * (
            len(self.convs)) + 2)
        pbar.set_description(stats_flag)

        xs = []
        for batch in subgraph_loader:
            batch = batch.to(self.device)
            x = x_all[batch.n_id.to(x_all.device)].to(self.device)
            x = F.relu(self.lin1(x))
            xs.append(x[:batch.batch_size].cpu())
            pbar.update(batch.batch_size)
        x_all = torch.cat(xs, dim=0)
        x_all = x_all_first = x_all.view(-1, 1, self.hidden_channels)
        # print("x_all shape is", x_all.shape)

        xs = []
        for batch in subgraph_loader:
            batch = batch.to(self.device)
            x, edge_index = batch.x, batch.edge_index
            _x_all = x_all[batch.n_id.to(x_all.device)].to(self.device)
            for i, conv in enumerate(self.convs):
                # print(">>>>", _x_all.device)
                x = F.relu(conv(_x_all, edge_index))
                x = x.view(-1, 1, self.hidden_channels)
                # print(">>>>", _x_all.device)
                # print(">>>>", x.device)
                _x_all = torch.cat([_x_all, x], dim=1)
                xs.append(_x_all[:batch.batch_size][:, -1].cpu())
                pbar.update(batch.batch_size)
        x_all = torch.cat(xs, dim=0)
        # x_all = x_all[:, -1]


        # for i, conv in enumerate(self.convs):
        #     xs = []
        #     # x_all_first_ = []
        #     for batch in subgraph_loader:
        #         batch = batch.to(self.device)
        #         x = x_all[batch.n_id.to(x_all.device)].to(self.device)
        #         x = F.relu(conv(x, batch.edge_index))
        #         x = x.view(-1, 1, self.hidden_channels)
        #         print("x shape is ", x.shape)
        #         # xs.append(torch.cat([x_all[batch.n_id].to(self.device), x], dim=1))
        #         xs.append(x[:batch.batch_size].cpu())
        #         pbar.update(batch.batch_size)
        #     x_all = torch.cat(xs, dim=0)
        #     print(x_all.shape)
        #     # x_all = torch.cat(xs, dim=0)
        # # x_all = x_all[:, -1]
        # x_all = x_all[:, -1]
        embedding_all = copy.copy(x_all)

        xs = []
        p_sim_list = []
        for batch in subgraph_loader:
            x = x_all[batch.n_id.to(x_all.device)].to(self.device)
            embedding = embedding_all[batch.n_id.to(x_all.device)].to(
                self.device)[:batch.batch_size]
            p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                                  training_embedding,
                                                                  training_y,
                                                                  None))
            p_sim_list.append(p_sim)
            x = self.lin2(x)
            xs.append(x[:batch.batch_size].cpu())
            pbar.update(batch.batch_size)
        x_all = torch.cat(xs, dim=0)
        p_sim_total = torch.cat(p_sim_list, dim=0)

        p_lc_total = self.m(x_all)
        final_label_distribution = self.eta * p_lc_total.to(self.device) + (
                1 - self.eta) * p_sim_total
        pbar.close()

        return final_label_distribution


class SSGC_Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 criterion,
                 k,
                 eta,
                 device):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        self.layer_1 = SSGConv(input_dim, hidden_dim, alpha=0.1, K=2)
        self.layer_2 = SSGConv(hidden_dim, num_classes, alpha=0.1, K=2)
        self.layers.append(self.layer_1)
        self.layers.append(self.layer_2)

        self.criterion = criterion
        self.k = k
        self.eta = eta
        if criterion == 'sigmoid':
            self.m = nn.Sigmoid()
        elif criterion == 'softmax':
            self.m = nn.LogSoftmax(dim=1)
        self.num_classes = num_classes
        self.device = device

    def pretrain_with_batch(self, data):

        x, edge_index = data.x, data.edge_index

        x = self.layer_1(x, edge_index)
        lc_x = self.layer_2(x, edge_index)

        lc_x = lc_x[:data.batch_size]
        p_lc = self.m(lc_x)

        return p_lc

    @torch.no_grad()
    def get_embedding(self, data):
        dic = {}
        dic['full_data'] = data
        x, edge_index = data.x, data.edge_index
        #y = data.y[:data.batch_size]
        y = data.y[data.train_mask][:data.batch_size]
        x = self.layer_1(x, edge_index)
        #embedding = x[:data.batch_size]
        embedding = x[data.train_mask][:data.batch_size]

        dic['train_embedding'] = embedding
        dic['train_label'] = y

        return dic

    @torch.no_grad()
    def decode_z_q(self, train_info):

        data = train_info['full_data']
        x, edge_index = data.x, data.edge_index

        y = data.y[:data.batch_size]
        x = self.layer_1(x, edge_index)
        lc_x = self.layer_2(x, edge_index)
        #lc_x = lc_x[:data.batch_size]
        lc_x = lc_x[data.train_mask][:data.batch_size]
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(train_info['z_q'],
                                                              train_info['z_q'],
                                                              #y,
                                                              data.y[data.train_mask],
                                                              train_info['neighbors_info']))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution

    def forward(self,
                data,
                neighbors_info=None):
        x, edge_index = data.x, data.edge_index

        y = data.y[:data.batch_size]
        x = self.layer_1(x, edge_index)
        #embedding = x[:data.batch_size]
        embedding = x[data.train_mask][:data.batch_size]
        lc_x = self.layer_2(x, edge_index)
        #lc_x = lc_x[:data.batch_size]
        lc_x = lc_x[data.train_mask][:data.batch_size]
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                              embedding,
                                                              #y,
                                                              data.y[data.train_mask],
                                                              neighbors_info))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution, embedding

    def generate_label_with_similar_nodes(self,
                                          embedding_a,
                                          embedding_b,
                                          b_y,
                                          neighbors_info=None):
        if self.training:
            mask = torch.ones(b_y.shape[0], b_y.shape[0])
            torch.diagonal(mask, 0).zero_()
        with torch.no_grad():
            if torch.is_tensor(neighbors_info):
                # print("|> [generate_label_with_similar_nodes] Using neighbors information")
                feature_similarity_matrix = neighbors_info
            else:
                # print("|> [generate_label_with_similar_nodes] Not Using neighbors information")
                feature_similarity_matrix = sim_matrix(embedding_a, embedding_b)
            feature_similarity_matrix = feature_similarity_matrix.cpu()
            if self.training:
                feature_similarity_matrix = feature_similarity_matrix * mask
            topk_similar_nodes_information = torch.topk(feature_similarity_matrix, self.k)
        fetch_similarity = torch.exp(topk_similar_nodes_information.values)
        fetch_similarity = fetch_similarity.to(self.device)
        fetch_label = []
        if self.criterion == 'sigmoid':
            one_hot_label = b_y
        elif self.criterion == 'softmax':
            one_hot_label = convert_to_one_hot_label(
                labels=b_y,
                num_classes=self.num_classes,
                batch_size=b_y.shape[0],
                device=self.device
            )
            one_hot_label = one_hot_label.to(self.device)

        for i in range(topk_similar_nodes_information.indices.shape[0]):
            fetch_label.append(
                torch.index_select(one_hot_label, 0, topk_similar_nodes_information.indices[i].to(self.device)))
        fetch_label = torch.stack(fetch_label)
        node_label_distribution = fetch_similarity.unsqueeze(-1) * fetch_label.to(self.device)
        total_fuse_distribution = node_label_distribution.sum(dim=1)

        return total_fuse_distribution

    def inference(self, x_all, subgraph_loader, training_embedding, training_y, stats_flag):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.layers))
        pbar.set_description(stats_flag)

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        xs = []
        p_sim_list = []
        for batch in subgraph_loader:
            batch = batch.to(self.device)
            x = x_all[batch.n_id.to(x_all.device)].to(self.device)
            x = self.layer_1(x, batch.edge_index)
            embedding = x[:batch.batch_size]
            p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                                  training_embedding,
                                                                  training_y,
                                                                  None))
            p_sim_list.append(p_sim)
            xs.append(x[:batch.batch_size].cpu())
            pbar.update(batch.batch_size)

        x_all = torch.cat(xs, dim=0)
        p_sim_total = torch.cat(p_sim_list, dim=0)

        xs = []
        for batch in subgraph_loader:
            batch = batch.to(self.device)
            x = x_all[batch.n_id.to(x_all.device)].to(self.device)
            x = self.layer_2(x, batch.edge_index)
            xs.append(x[:batch.batch_size].cpu())
            pbar.update(batch.batch_size)
        x_all = torch.cat(xs, dim=0)

        p_lc_total = self.m(x_all)
        final_label_distribution = self.eta * p_lc_total.to(self.device) + (1 - self.eta) * p_sim_total
        pbar.close()

        return final_label_distribution


class FAGCN_Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 criterion,
                 k,
                 eta,
                 device,
                 n_layers=1,
                 eps=0.2,
                 dropout=0.5):
        super().__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(input_dim, hidden_dim))
        self.lins.append(nn.Linear(hidden_dim, num_classes))
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                FAConv(hidden_dim, dropout=dropout))
        self.dropout = dropout

        self.criterion = criterion
        self.k = k
        self.eta = eta
        if criterion == 'sigmoid':
            self.m = nn.Sigmoid()
        elif criterion == 'softmax':
            self.m = nn.LogSoftmax(dim=1)
        self.num_classes = num_classes
        self.device = device
        self.eps = eps

    def pretrain_with_batch(self, data):
        """
        预训练环节
        :param data:
        :return:
        """
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        x_0 = F.dropout(x_0, self.dropout, training=self.training)

        for conv in self.convs:
            h = conv(x, x_0, edge_index)
            x = self.eps * x_0 + h

        lc_x = self.lins[1](x)
        lc_x = lc_x[:data.batch_size]

        p_lc = self.m(lc_x)

        return p_lc

    @torch.no_grad()
    def get_embedding(self, data):
        dic = {}
        dic['full_data'] = data
        #y = data.y[:data.batch_size]
        y = data.y[data.train_mask][:data.batch_size]

        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        x_0 = F.dropout(x_0, self.dropout, training=self.training)

        for conv in self.convs:
            h = conv(x, x_0, edge_index)
            x = self.eps * x_0 + h
            x = x.relu()

        #embedding = x[:data.batch_size]
        embedding = x[data.train_mask][:data.batch_size]

        dic['train_embedding'] = embedding
        dic['train_label'] = y

        return dic

    @torch.no_grad()
    def decode_z_q(self, train_info):

        data = train_info['full_data']
        y = data.y[:data.batch_size]

        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        x_0 = F.dropout(x_0, self.dropout, training=self.training)

        for conv in self.convs:
            h = conv(x, x_0, edge_index)
            x = self.eps * x_0 + h
            x = x.relu()

        lc_x = self.lins[1](x)
        #lc_x = lc_x[:data.batch_size]
        lc_x = lc_x[data.train_mask][:data.batch_size]

        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(train_info['z_q'],
                                                              train_info['z_q'],
                                                              #y,
                                                              data.y[data.train_mask],
                                                              train_info['neighbors_info']))
        # print(p_sim)

        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution

    def forward(self,
                data,
                mask,
                neighbors_info=None):

        y = data.y[:data.batch_size]

        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        x_0 = F.dropout(x_0, self.dropout, training=self.training)

        for conv in self.convs:
            h = conv(x, x_0, edge_index)
            x = self.eps * x_0 + h
            x = x.relu()

        #embedding = x[:data.batch_size]
        embedding = x[data.train_mask][:data.batch_size]
        lc_x = self.lins[1](x)
        #lc_x = lc_x[:data.batch_size]
        lc_x = lc_x[data.train_mask][:data.batch_size]
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                              embedding,
                                                              #y,
                                                              data.y[data.train_mask],
                                                              neighbors_info))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution, embedding

    def generate_label_with_similar_nodes(self,
                                          embedding_a,
                                          embedding_b,
                                          b_y,
                                          neighbors_info=None):
        if self.training:
            mask = torch.ones(b_y.shape[0], b_y.shape[0])
            torch.diagonal(mask, 0).zero_()
        with torch.no_grad():
            if torch.is_tensor(neighbors_info):
                # print("|> [generate_label_with_similar_nodes] Using neighbors information")
                feature_similarity_matrix = neighbors_info
            else:
                # print("|> [generate_label_with_similar_nodes] Not Using neighbors information")
                feature_similarity_matrix = sim_matrix(embedding_a, embedding_b)
            feature_similarity_matrix = feature_similarity_matrix.cpu()
            if self.training:
                feature_similarity_matrix = feature_similarity_matrix * mask
            topk_similar_nodes_information = torch.topk(
                feature_similarity_matrix, self.k)
        fetch_similarity = torch.exp(topk_similar_nodes_information.values)
        fetch_similarity = fetch_similarity.to(self.device)
        fetch_label = []
        if self.criterion == 'sigmoid':
            one_hot_label = b_y
        elif self.criterion == 'softmax':
            one_hot_label = convert_to_one_hot_label(
                labels=b_y,
                num_classes=self.num_classes,
                batch_size=b_y.shape[0],
                device=self.device
            )
            one_hot_label = one_hot_label.to(self.device)

        for i in range(topk_similar_nodes_information.indices.shape[0]):
            fetch_label.append(
                torch.index_select(one_hot_label, 0,
                                   topk_similar_nodes_information.indices[i].to(
                                       self.device)))
        fetch_label = torch.stack(fetch_label)
        node_label_distribution = fetch_similarity.unsqueeze(
            -1) * fetch_label.to(self.device)
        total_fuse_distribution = node_label_distribution.sum(dim=1)

        return total_fuse_distribution

    def inference(self,
                  x_all,
                  subgraph_loader,
                  training_embedding,
                  training_y,
                  stats_flag):

        pbar = tqdm(total=len(subgraph_loader.dataset) * (
                len(self.convs) + len(self.lins)))
        pbar.set_description(stats_flag)

        xs = []
        for batch in subgraph_loader:
            x = x_all[batch.n_id.to(x_all.device)].to(self.device)
            x = self.lins[0](x).relu()
            xs.append(x[:batch.batch_size].cpu())
            pbar.update(batch.batch_size)
        x_all = x_all_first = torch.cat(xs, dim=0)

        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(self.device)
                x_0 = x_all_first[batch.n_id.to(x_all.device)].to(self.device)
                h = conv(x, x_0, batch.edge_index.to(self.device))
                x = self.eps * x_0 + h
                x = x.relu()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)

        embedding_all = copy.copy(x_all)

        xs = []
        p_sim_list = []
        for batch in subgraph_loader:
            x = x_all[batch.n_id.to(x_all.device)].to(self.device)
            embedding = embedding_all[batch.n_id.to(x_all.device)].to(
                self.device)[:batch.batch_size]
            # print(embedding[:])
            p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                                  training_embedding,
                                                                  training_y,
                                                                  None))
            p_sim_list.append(p_sim)
            x = self.lins[1](x)
            xs.append(x[:batch.batch_size].cpu())
            pbar.update(batch.batch_size)
        x_all = torch.cat(xs, dim=0)
        p_sim_total = torch.cat(p_sim_list, dim=0)

        p_lc_total = self.m(x_all)
        final_label_distribution = self.eta * p_lc_total.to(self.device) + (
                1 - self.eta) * p_sim_total
        pbar.close()

        return final_label_distribution