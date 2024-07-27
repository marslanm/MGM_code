
import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, SGConv, GCN2Conv
from utils import convert_to_one_hot_label
import math
from tqdm import tqdm
from gnn_encoder import GraphSAGE_Encoder, \
    GAT_Encoder, \
    GCN_Encoder, \
    SGC_Encoder, \
    GCNII_Encoder, \
    GATv2Conv_Encoder, \
    FILM_Encoder, \
    DNA_Encoder, \
    SSGC_Encoder, \
    FAGCN_Encoder



def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class NodeReprVAE(nn.Module):

    def __init__(self,
                 model_name,
                 hidden_dim1,
                 hidden_dim2,
                 num_classes,
                 normalize,
                 delta,
                 k,
                 sigma,
                 criterion,
                 device):
        super(NodeReprVAE, self).__init__()
        self.logstd_p_z = None
        self.mu_p_z = None
        self.logstd_q_z = None
        self.mu_q_z = None
        if 'gat' in model_name:
            n = 4
        else:
            n = 1
        self.gnn_mean_q_z = nn.Linear(hidden_dim1*n, hidden_dim2)
        self.gnn_logstd_q_z = nn.Linear(hidden_dim1*n, hidden_dim2)
        self.gnn_mean_p_z = nn.Linear(hidden_dim1*n, hidden_dim2)
        self.gnn_logstd_p_z = nn.Linear(hidden_dim1*n, hidden_dim2)
        self.linear_layer = nn.Linear(hidden_dim1*n, hidden_dim1*n)


        self.delta = delta
        self.delta = delta
        self.sigma = sigma
        self.criterion = criterion
        self.device = device
        self.k = k
        self.num_classes = num_classes


    def encode(self, gnn_embedding, sim_info):
        # print(gnn_embedding.shape)
        # print(sim_info['most_similar_embedding'].shape)
        # Equation 23
        self.mu_q_z = self.gnn_mean_q_z(
            gnn_embedding + self.delta * self.linear_layer(sim_info['most_similar_embedding']))
        self.logstd_q_z = self.gnn_logstd_q_z(
            gnn_embedding + self.delta * self.linear_layer(sim_info['most_similar_embedding']))

        # Equation 8
        self.mu_p_z = self.gnn_mean_p_z(gnn_embedding)
        self.logstd_p_z = self.gnn_logstd_p_z(gnn_embedding)

        return self.mu_q_z, self.logstd_q_z, self.mu_p_z, self.logstd_p_z


    def reparameterize(self, mu, logstd):

        if self.training:
            std = torch.exp(self.sigma * logstd)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu


    def forward(self, train_info, sim_info):
        mu_q_z, logstd_q_z, mu_p_z, logstd_p_z = self.encode(train_info['train_embedding'], sim_info)
        z_q = self.reparameterize(mu_q_z, logstd_q_z)
        train_info['z_q'] = z_q
        return [train_info, z_q, mu_q_z, logstd_q_z]


    def compute_KL(self):
        kl = (self.logstd_p_z - self.logstd_q_z) + (self.logstd_q_z.exp() ** 2 + (self.mu_p_z - self.mu_q_z) ** 2) / (
                2 * self.logstd_p_z.exp() ** 2) - 0.5
        return -torch.mean(torch.sum(kl, 1))


class SimilarNodeDistribution(object):

    def __init__(self, k, device, criterion, num_classes, training_status):
        self.k = k
        self.device = device
        self.criterion = criterion
        self.status = criterion
        self.num_classes = num_classes
        self.training = training_status

    def transform_label(self,
                        label):

        if self.criterion == 'sigmoid':
            one_hot_label = label
            return one_hot_label

        elif self.criterion == 'softmax':
            one_hot_label = convert_to_one_hot_label(
                labels=label,
                num_classes=self.num_classes,
                batch_size=label.shape[0],
                device=self.device
            )
            return one_hot_label

    def fetch_sim_info(self,
                       embedding_a,
                       a_y,
                       embedding_b,
                       b_y,
                       train_batch_id,
                       neighbors_info=None):
        #breakpoint()
        one_hot_label_a = self.transform_label(a_y)
        one_hot_label_b = self.transform_label(b_y)

        if self.training:
            mask = torch.ones(one_hot_label_a.shape[0], one_hot_label_a.shape[0])
            torch.diagonal(mask, 0).zero_()

        # Equation 22
        embedding_combine_a = torch.cat((embedding_a, one_hot_label_a), dim=1)
        embedding_combine_b = torch.cat((embedding_b, one_hot_label_b), dim=1)

        with torch.no_grad():
            if torch.is_tensor(neighbors_info):
                # print("|> [fetch_sim_info] Using neighborhoods information.")
                similarity_matrix = neighbors_info
            else:
                # print("|> [fetch_sim_info] Not Using neighborhoods information.")
                similarity_matrix = sim_matrix(embedding_combine_a, embedding_combine_b)
            similarity_matrix = similarity_matrix.cpu()
            if self.training:
                similarity_matrix = similarity_matrix * mask

            gamma_weights = F.softmax(similarity_matrix, dim=1)
            # Equation 21
            #breakpoint()
            topk_similar_nodes_information = torch.topk(gamma_weights, self.k)

        dic = {}

        batch_id_top_k = []
        for i in range(topk_similar_nodes_information.indices.shape[0]):
            batch_id_top_k.append(
                torch.index_select(train_batch_id, 0,
                                   topk_similar_nodes_information.indices[i].to(self.device))
            )
        batch_id_top_k = torch.stack(batch_id_top_k)

        similarity_values = []
        for i in range(topk_similar_nodes_information.indices.shape[0]):
            similarity_values.append(
                torch.index_select(similarity_matrix[i], 0,
                                   topk_similar_nodes_information.indices[i]))
        similarity_values = torch.stack(similarity_values)

        similarity_labels = []
        for i in range(topk_similar_nodes_information.indices.shape[0]):
            similarity_labels.append(
                torch.index_select(one_hot_label_b, 0,
                                   topk_similar_nodes_information.indices[i].to(self.device)))

        similarity_labels = torch.stack(similarity_labels)

        similarity_embedding = []
        for i in range(topk_similar_nodes_information.indices.shape[0]):
            similarity_embedding.append(
                torch.index_select(embedding_b, 0,
                                   topk_similar_nodes_information.indices[i].to(self.device)))

        similarity_embedding = torch.stack(similarity_embedding)

        # Equation 25
        # Indicate function
        scalar_output = torch.sum(one_hot_label_a.repeat(self.k, 1, 1).permute(1, 0, 2) == similarity_labels,
                                  dim=-1)
        scalar_output = scalar_output / one_hot_label_a.shape[1]
        #         print(scalar_output)
        #         print(topk_similar_nodes_information.values)
        b_uv = F.softmax(scalar_output * topk_similar_nodes_information.values.to(self.device), dim=1)

        # Equation 24
        most_similar_embedding = b_uv.unsqueeze(-1) * similarity_embedding
        most_similar_embedding = torch.sum(most_similar_embedding, dim=1)

        dic['tu_indexes'] = topk_similar_nodes_information.indices
        dic['tu_indexes_in_global'] = batch_id_top_k
        dic['gamma_weights'] = topk_similar_nodes_information.values
        dic['similarity_values'] = similarity_values
        dic['similarity_labels'] = similarity_labels
        dic['similarity_embedding'] = similarity_embedding
        dic['most_similar_embedding'] = most_similar_embedding.float()
        return dic


class MGM(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 normalize,
                 nodes_numbers,
                 model_name,
                 criterion,
                 k,
                 eta,
                 device,
                 alpha=0.1):
        super(MGM, self).__init__()

        self.k = k
        self.eta = eta
        self.device = device
        self.criterion = criterion
        self.num_classes = num_classes

        self.nodes_numbers = nodes_numbers
        # Dirichlet vae_prior
        self.alpha = torch.tensor([alpha] * self.nodes_numbers).to(self.device)
        # Dirichlet posterior
        self.lambda_ = nn.Parameter(torch.ones(self.nodes_numbers).fill_(alpha))
        self.lambda_.requires_grad = False
        self.alpha_stats = self.digamma_stats1(self.alpha)
        self.forget_rate = 0.8
        self.decay_rate = 1.0
        self._num_updates = 0
        self.lambda_stats_sum = nn.Parameter(torch.ones(self.nodes_numbers).fill_(alpha))

        self.embedding_encoder = self.construct_embedding_model(
            input_dim,
            hidden_dim,
            num_classes,
            normalize,
            model_name,
            criterion,
            k,
            eta,
            device)

        self.similar_node_handler = SimilarNodeDistribution(
            k=self.k,
            device=self.device,
            criterion=self.criterion,
            num_classes=self.num_classes,
            training_status=self.training)

        self.node_vae = NodeReprVAE(
            model_name=model_name,
            hidden_dim1=hidden_dim,
            hidden_dim2=64,
            num_classes=num_classes,
            normalize=True,
            delta=0.2,
            k=k,
            sigma=0.1,
            criterion=criterion,
            device=device
        )

        self.training_embedding_memory = None
        self.train_label_memory = None
        self.batch_size = None

    def construct_embedding_model(self,
                                  input_dim,
                                  hidden_dim,
                                  num_classes,
                                  normalize,
                                  model_name,
                                  criterion,
                                  k,
                                  eta,
                                  device):
        # print(model_name)
        if model_name == 'graphsage':
            embedding_encoder = GraphSAGE_Encoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                criterion=criterion,
                k=k,
                eta=eta,
                device=device
            )

        elif model_name == 'gat':
            embedding_encoder = GAT_Encoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                criterion=criterion,
                k=k,
                eta=eta,
                device=device
            )

        elif model_name == 'gatv2':
            embedding_encoder = GATv2Conv_Encoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                criterion=criterion,
                k=k,
                eta=eta,
                device=device
            )

        elif model_name == 'gcn':
            embedding_encoder = GCN_Encoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                criterion=criterion,
                k=k,
                eta=eta,
                device=device
            )

        elif model_name == 'sgc':
            embedding_encoder = SGC_Encoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                criterion=criterion,
                k=k,
                eta=eta,
                device=device
            )

        elif model_name == 'gcnii':
            print("|> Here.")
            embedding_encoder = GCNII_Encoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                criterion=criterion,
                k=k,
                eta=eta,
                device=device
            )

        elif model_name == 'film':
            embedding_encoder = FILM_Encoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                criterion=criterion,
                k=k,
                eta=eta,
                device=device
            )

        elif model_name == 'dna':
            embedding_encoder = DNA_Encoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                criterion=criterion,
                k=k,
                eta=eta,
                device=device
            )

        elif model_name == 'ssgc':
            embedding_encoder = SSGC_Encoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                criterion=criterion,
                k=k,
                eta=eta,
                device=device
            )

        elif model_name == 'fagcn':
            embedding_encoder = FAGCN_Encoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                criterion=criterion,
                k=k,
                eta=eta,
                device=device
            )



        return embedding_encoder

    def pretrain_forward(self, data):

        return self.embedding_encoder.pretrain_with_batch(data)

    def m_step_forward(self,
                       data,
                       neighbors_info=None):

        return self.embedding_encoder(data,
                                      neighbors_info)

    def get_training_embedding_memory(self,
                                     data,
                                     neighbors_info):

        self.embedding_encoder.eval()
        train_info = self.embedding_encoder.get_embedding(data)
        self.training_embedding_memory = train_info['train_embedding']
        self.train_label_memory = train_info['train_label']
        #train_info['train_batch_id'] = data.n_id[:data.batch_size]
        train_info['train_batch_id'] = data.n_id[data.train_mask][:data.batch_size]
        train_info['neighbors_info'] = neighbors_info
        self.embedding_encoder.train()

        return train_info

    def get_sim_info(self,
                     embedding_a,
                     a_y,
                     embedding_b,
                     b_y,
                     train_batch_id,
                     neighbors_info=None):

        return self.similar_node_handler.fetch_sim_info(
            embedding_a=embedding_a,
            embedding_b=embedding_b,
            a_y=a_y,
            b_y=b_y,
            train_batch_id=train_batch_id,
            neighbors_info=neighbors_info
        )

    def reconstruct_z_q(self, train_info, sim_info):

        _train_info, _z_q, _mu_q_z, _logstd_q_z = self.node_vae(train_info=train_info,
                                                                sim_info=sim_info)
#         print("|> _train_info is ", _train_info)
#         print("|> _z_q is is ", _z_q)
#         print("|> _mu_q_z is ", _mu_q_z)
#         print("|> _logstd_q_z is ", _logstd_q_z)
        final_label_distribution = self.embedding_encoder.decode_z_q(_train_info)

        return final_label_distribution

    def update_lambda(self, value):
        self.lambda_.copy_(value)

    def distributed_update_lambda(self,
                                  lambda_stats_sum,
                                  num_of_nodes,
                                  update_num):

        """
        num_of_nodes: Batch size

        """

        forget = math.pow(update_num + self.decay_rate, -self.forget_rate)
        new_lambda = self.alpha + lambda_stats_sum / num_of_nodes * self.nodes_numbers

        # Update the Dirichlet posterior
        new_lambda = (1 - forget) * self.lambda_ + forget * new_lambda
        self.update_lambda(new_lambda)

    def digamma_stats1(self, alpha):
        return torch.lgamma(alpha.sum()) - torch.lgamma(alpha).sum()

    def digamma_stats2(self, alpha1, alpha2):
        return ((alpha1 - 1.) * (torch.digamma(alpha2) - torch.digamma(alpha2.sum()))).sum()

    def update_lambda_stats_sum(self, sim_info):

        with torch.no_grad():
            lambda_stats_sum_i = nn.Parameter(torch.ones(self.nodes_numbers).fill_(0)).to(self.device)
            self.batch_size = sim_info['similarity_labels'].shape[0]
            #breakpoint()
            lambda_stats_sum_i = lambda_stats_sum_i.repeat(self.batch_size, 1).scatter(-1,sim_info['tu_indexes_in_global'],self.k * sim_info['gamma_weights'].to(self.device))
            lambda_stats_sum_i = sum(lambda_stats_sum_i)
            self.lambda_stats_sum += lambda_stats_sum_i
            

    def compute_kl_theta(self):
        # E_{q(\theta; \lambda)}[log p(\theta) - \log q(\theta)] / (number of all training samples)
        if self.training:
            term1 = self.alpha_stats + self.digamma_stats2(self.alpha, self.lambda_) \
                    - (self.digamma_stats1(self.lambda_) +
                       self.digamma_stats2(self.lambda_, self.lambda_))
            term1 = term1 / self.nodes_numbers
        else:
            term1 = 0.

        return -term1