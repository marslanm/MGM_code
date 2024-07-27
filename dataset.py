import torch
import pandas as pd
from torch_geometric.data import Data

def data_load(path="./data/NMP/", dataset="NMP"):
    print('Loading {} dataset...'.format(dataset))
    fact_dir = "Fact/"
    bias_dir = "Bias/"
    fact_file = "ACL_level_3_fact.csv"
    bias_file = "ACL_level_3_bias.csv"
    edge_file = "Edge_level_3.csv"

    fact_data = path + fact_dir + fact_file
    bias_data = path + bias_dir + bias_file
    edge_path = path + edge_file

    edge_df = pd.read_csv(edge_path)
    #nodes_df = pd.read_csv(fact_data, index_col=0)
    nodes_df = pd.read_csv(bias_data, index_col=0)

    data = create_graph(list(zip(edge_df['source'], edge_df['target'])),
                            list(zip(
                                   #nodes_df['alexa_rank'],
                                   nodes_df['daily_pageviews_per_visitor'],
                                   nodes_df['daily_time_on_site'],
                                   #nodes_df['total_sites_linking_in'],
                                   nodes_df['bounce_rate'],
                                   nodes_df['normalized_alexa_rank'],
                                   nodes_df['normalized_total_sites_linked_in']
                                   )),
                          nodes_df['class_factorized'], nodes_df['train_mask'],
                          nodes_df['test_mask'], nodes_df['unlabel_mask'])

    return data

def create_graph(edges, features, labels, train_mask, test_mask, unlabel_mask):
    edge_index = torch.tensor(edges, dtype=torch.long)
    # labels = np.array(labels)
    # labels = labels[~np.isnan(labels)]
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
    unlabel_mask = torch.tensor(unlabel_mask, dtype=torch.bool)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), train_mask=train_mask, unlabel_mask=unlabel_mask,
                test_mask=test_mask, y=y)
    return data

