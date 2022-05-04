import torch as th
from torch import nn
import torch
import numpy as np
from dgl import function as fn
from dgl.base import DGLError
from scipy.special import factorial


class EGConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 k=1,
                 cached=False,
                 bias=True,
                 norm=None,
                 allow_zero_in_degree=False):
        super(EGConv, self).__init__()
        self.in_feats=in_feats
        self.out_feats=out_feats
        self.fc = nn.Linear(in_feats, out_feats, bias=bias)
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self._lambda =nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / np.sqrt(self.out_feats)
        self.fc.weight.data.uniform_(-stdv, stdv)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

        stdvk = 1. / np.sqrt(self._k)
        self._lambda.data.uniform_(-stdvk,stdvk)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):

        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if self._cached_h is not None:
                feat_list = self._cached_h
                result = torch.zeros(feat_list[0].shape[0], self.out_feats).to(feat_list[0].device)
                for i, k_feat in enumerate(feat_list):
                    result += self.fc(k_feat * (self._lambda.pow(i) / factorial(i)))
            else:
                feat_list = []

                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)

                feat_list.append(feat.float())
                for i in range(self._k):
                    feat = feat * norm
                    feat=feat.float()
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'),
                                     fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm
                    feat_list.append(feat)

                result = torch.zeros(feat_list[0].shape[0],self.out_feats).to(feat_list[0].device)
                for i,k_feat in enumerate(feat_list):
                    result += self.fc(k_feat*(self._lambda.pow(i)/factorial(i)))


                if self.norm is not None:
                    result = self.norm(result)

                # cache feature
                if self._cached:
                    self._cached_h = feat_list
            return result











