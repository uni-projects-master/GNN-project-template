import os
import sys
import sys
path='C:/Users/solma/OneDrive/Documents/GitHub/Empowering-Simple-Graph-Convolutional-Networks'
sys.path.append(path)
#sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import torch
import dgl
from dgl import DGLGraph

from dataReader_utils.WikiCS import load_graph_data



def DGLDatasetReader(dataset_name,self_loops,device=None):

    dataset = load_data(dataset_name,self_loops)
    
    g = dataset[0]
    feat = g.ndata['feat']
    
    n_classes = dataset.num_classes

    # get node feature
    feat = g.ndata['feat']

    # get data split
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    # get labels
    labels = g.ndata['label']
    
    # add self loop
    '''if self_loops:
        # g.add_edges(g.nodes(), g.nodes())
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)'''
        

    return g,torch.FloatTensor(feat),torch.LongTensor(labels),n_classes,\
           torch.ByteTensor(train_mask),torch.ByteTensor(test_mask),torch.ByteTensor(val_mask)


def WikiCS_loader(self_loops=False,device=None):
    data = load_graph_data.load(self_loop=self_loops)
    g = data.graph.to(device)
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    norm = norm.to(device)
    g.ndata['norm'] = norm.unsqueeze(1)
    # add self loop
    if self_loops:
        # g.add_edges(g.nodes(), g.nodes())
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    return g, torch.FloatTensor(data.features), torch.LongTensor(data.labels), max(data.labels)+1, \
           data.train_masks, data.test_mask, data.val_masks


from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from sklearn import preprocessing
def ogbn_arxiv_reader(device='cpu',data_root_dir="/Users/lpasa/Dataset/"):
    dataset = 'ogbn-arxiv'
    dataset = DglNodePropPredDataset(name=dataset,root=data_root_dir)
    g, labels = dataset[0]  # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
    srcs, dsts = g.all_edges()
    g.add_edges(dsts, srcs)
    features = g.ndata['feat']
    min_max_scaler = preprocessing.MinMaxScaler()
    feat = min_max_scaler.fit_transform(features)
    # feat_smooth_matrix = calc_feat_smooth(adj, feat)
    n_classes = (labels.max() + 1).item()

    g = g.remove_self_loop().add_self_loop()
    old_g = DGLGraph(g.to_networkx())
    old_g.readonly()

    evaluator = Evaluator(name="ogbn-arxiv")
    split_idxdx = dataset.get_idx_split()
    idx_train, idx_val, idx_test = split_idxdx["train"], split_idxdx["valid"], split_idxdx["test"]

    return g.to(device), torch.tensor(feat, dtype=torch.float32).to(device), labels[:,0].to(device), n_classes, idx_train.to(device), idx_val.to(device), idx_test.to(device), evaluator



def load_data(dataset_name,self_loops):
    if dataset_name == 'cora':
        return dgl.data.CoraGraphDataset()
    elif dataset_name == 'citeseer':
        return dgl.data.CiteseerGraphDataset
    elif dataset_name== 'pubmed':
        return dgl.data.PubmedGraphDataset
    elif dataset_name== "PPI":
        return dgl.data.PPIDataset
    elif dataset_name== "WikiCS":
        return WikiCS_loader(self_loops)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_name))



from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from sklearn import preprocessing

def ogbn_arxiv_reader(device='cpu',data_root_dir="/Users/lpasa/Dataset/"):
    dataset = 'ogbn-arxiv'
    dataset = DglNodePropPredDataset(name=dataset,root=data_root_dir)
    g, labels = dataset[0]  # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
    srcs, dsts = g.all_edges()
    g.add_edges(dsts, srcs)
    features = g.ndata['feat']
    min_max_scaler = preprocessing.MinMaxScaler()
    feat = min_max_scaler.fit_transform(features)
    # feat_smooth_matrix = calc_feat_smooth(adj, feat)
    n_classes = (labels.max() + 1).item()

    g = g.remove_self_loop().add_self_loop()
    old_g = DGLGraph(g.to_networkx())
    old_g.readonly()

    evaluator = Evaluator(name="ogbn-arxiv")
    split_idxdx = dataset.get_idx_split()
    idx_train, idx_val, idx_test = split_idxdx["train"], split_idxdx["valid"], split_idxdx["test"]

    return g.to(device), torch.tensor(feat, dtype=torch.float32).to(device), labels[:,0].to(device), n_classes, idx_train.to(device), idx_val.to(device), idx_test.to(device), evaluator

