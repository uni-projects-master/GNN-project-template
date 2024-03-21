from torch import nn
import torch


class LGNetwork(nn.Module):
    def __init__(self,
                 lg_list,
                 in_feats,
                 h_feats,
                 n_classes,
                 lg_node_list,
                 inc_type,
                 num_layers,
                 activation,
                 dropout,
                 convLayer,
                 out_fun=nn.Softmax(dim=1),
                 device=None,
                 norm=None,
                 bias=False):
        super(LGNetwork, self).__init__()

        self.lg_list = lg_list
        self.lg_node_list = lg_node_list
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.layers = nn.ModuleList()
        self.n_classes = n_classes
        self.in_feats = in_feats
        self.out_fun = out_fun
        self.num_layer = num_layers

        self.layers.append(convLayer(in_feats, h_feats, lg_list, inc_type, dropout, activation))
        for i in range(num_layers):
            self.layers.append(convLayer(h_feats, h_feats, lg_list, inc_type, dropout, activation))

        self.outLayer = nn.Linear(h_feats, n_classes)

    def reset_lg_feats(self, feat_0):
        feat_list = []
        for i in range(len(self.lg_node_list)):
            if i == 0:
                feat_list.append(feat_0)
            else:
                feat_list.append(torch.zeros(self.lg_node_list[i], self.in_feats))
        return feat_list

    def forward(self, features):
        lg_h = self.reset_lg_feats(features)
        for layer in self.layers:
            lg_h = layer(lg_h)
        h = self.outLayer(lg_h[0])

        return self.out_fun(h), h
