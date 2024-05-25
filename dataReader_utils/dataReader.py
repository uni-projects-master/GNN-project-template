import os
import sys
import dgl
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))


def DGLDatasetReader(dataset_name, backtrack=True, lg_level=1, problem_type='fully-supervised', self_loops=True, device=None):
    dataset = load_data(dataset_name)

    g = dataset[0].to(device)

    n_classes = dataset.num_classes
    feat = g.ndata['feat']
    n_feats = feat.shape[1]
    # get labels
    labels = g.ndata['label']
    num_nodes = g.num_nodes()

    # Semi-supervised data split
    if problem_type == "semi-supervised":
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']

    else:

        num_true_values = [int(0.8 * num_nodes), int(0.1 * num_nodes), int(0.1 * num_nodes)]

        tensor1 = np.zeros(num_nodes, dtype=bool)
        tensor2 = np.zeros(num_nodes, dtype=bool)
        tensor3 = np.zeros(num_nodes, dtype=bool)

        indices = np.random.choice(num_nodes, sum(num_true_values), replace=False)

        start_index = 0
        for i, num_true in enumerate(num_true_values):
            end_index = start_index + num_true
            tensor = [tensor1, tensor2, tensor3][i]
            tensor.flat[indices[start_index:end_index]] = True
            start_index = end_index

        tensor2 = tensor2 & ~tensor1
        tensor3 = tensor3 & ~tensor1 & ~tensor2

        train_mask = torch.from_numpy(tensor1)
        val_mask = torch.from_numpy(tensor2)
        test_mask = torch.from_numpy(tensor3)

    # add self loop
    if self_loops:
        g = dgl.add_self_loop(g)

    line_graphs, lg_node_list = create_line_graphs(g, n_feats, lg_level, backtrack, device)

    return line_graphs, feat, labels, n_classes, n_feats, lg_node_list, train_mask, test_mask, val_mask


def create_line_graphs(g, num_feats, lg_level, backtrack, device):
    graphs = []
    num_node_list = []
    graphs.append(g)
    num_node_list.append(g.num_nodes())
    current_g = g
    for t in range(0, lg_level):
        lg = current_g.line_graph(backtracking=backtrack).to(device)
        graphs.append(lg)
        num_node_list.append(lg.num_nodes())
        current_g = lg
    return graphs, num_node_list


def load_data(dataset_name):
    if dataset_name == 'Cora':
        return dgl.data.CoraGraphDataset()
    elif dataset_name == 'Citeseer' or dataset_name == 'citeseer':
        return dgl.data.CiteseerGraphDataset()
    elif dataset_name == 'Pubmed':
        return dgl.data.PubmedGraphDataset()
    elif dataset_name == "PPI":
        return dgl.data.PPIDataset
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_name))

