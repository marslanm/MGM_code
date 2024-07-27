# MGM: Global Understanding of Audience Overlap Graphs for Predicting the Factuality and the Bias of News Media

## Installation
## Installation
The codebase has been tested on Ubuntu 20.04.2 LTS using Python 3.8. To set up the environment and install the necessary dependencies, please follow the steps outlined below.


```shell
conda env create -f environment.yml
```

## Implementation Details

MGM can be equipped with various GNN models:

- GCN
- GraphSAGE
- GAT
- SGC
- DNA
- FILM
- FAGCN
- GATv2Conv



## Data 

Data files are available in the data directory with the name of fact and bias. dataset.py is used to load the bias data. dataset_fact.py is used to load the data for factuality task.

## MGM Training 

- To replicate our experimental results for the Factuality dataset, please use the following command to train the MGM based models:

```bash
python nmp_fact.py --cuda_id 0 --model [gcn/graphsage/gat/sgc/dna/gcnii/film/ssgc/fagcn/gatv2] --criterion sigmoid --hidden_dim [16,32,64..] --log_dir ./your_log --k 3 --eta [0.5,0.6,0.7,0.8,0.9,1] --val_test_batch_size 2 --epochs 50 --run_times 5 --normalize True --gnn_lr 0.001 --vae_lr 0.0001 --sim_function feature_base
```

- To replicate our experimental results for the Bias dataset, please use the following command to train the MGM based models:

```bash
python nmp_bias.py --cuda_id 0 --model [gcn/graphsage/gat/sgc/dna/gcnii/film/ssgc/fagcn/gatv2] --criterion sigmoid --hidden_dim [16,32,64..] --log_dir ./your_log --k 3 --eta [0.5,0.6,0.7,0.8,0.9,1] --val_test_batch_size 2 --epochs 50 --run_times 5 --normalize True --gnn_lr 0.001 --vae_lr 0.0001 --sim_function feature_base
```

## Descriptions for command arguments:
hidden_dim: The dimensionality of the hidden layer, matching the baseline's configuration as described in the main paper.

K: The number of globally similar nodes to consider.

eta: A hyperparameter controlling the balance between local and global information.

    eta = 1: Relies solely on local information, similar to the baseline models.
    eta = 0.5: Equally weighs local and global information using the MGM approach.
