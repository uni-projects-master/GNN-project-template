import torch as th
from torch import nn
import torch
from dgl import function as fn
from dgl.base import DGLError
from utils.utils_method import IdFun
from conv.EGConv import EGConv

class hLGConv(EGConv):
    def __init__(self,
                 in_feats,
                 out_feats,
                 k=1,
                 cached=False,
                 bias=True,
                 norm=None,
                 allow_zero_in_degree=False):
        super(hLGConv, self).__init__(in_feats,
                                      out_feats,
                                      k,
                                      cached,
                                      bias,
                                      norm,
                                      allow_zero_in_degree)

        self._lambada_fun = nn.Linear(in_feats + out_feats, in_feats)
        self._lambada_act = IdFun()

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
                feat_km1 = feat_list[0]
                for i, k_feat in enumerate(feat_list):
                    _dLambda = self._lambada_act(self._lambada_fun(torch.cat([feat_km1, result], dim=1)))
                    result += self.fc(k_feat * _dLambda)
                    feat_km1 = k_feat

            else:
                feat_list = []

                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)

                feat_list.append(feat.float())
                # compute (D^-1 A^k D)^k X
                for i in range(self._k):
                    # norm = D^-1/2
                    feat = feat * norm
                    feat = feat.float()
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'),
                                     fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    # compute (D^-1 A^k D)^k X
                    feat = feat * norm
                    feat_list.append(feat)

                result = torch.zeros(feat_list[0].shape[0], self.out_feats).to(feat_list[0].device)
                feat_km1 = feat_list[0]
                for i, k_feat in enumerate(feat_list):
                    _dLambda = self._lambada_act(self._lambada_fun(torch.cat([feat_km1, result], dim=1)))
                    result += self.fc(k_feat * _dLambda)
                    feat_km1 = k_feat

                if self.norm is not None:
                    result = self.norm(result)

                # cache feature
                if self._cached:
                    self._cached_h = feat_list
            return result

from utils.utils_method import SimpleMultiLayerNN
from math import floor

class hLGConv_deep(hLGConv):

    def __init__(self,
                 in_feats,
                 out_feats,
                 k=1,
                 cached=False,
                 bias=True,
                 norm=None,
                 allow_zero_in_degree=False):
        super(hLGConv, self).__init__(in_feats,
                                      out_feats,
                                      k,
                                      cached,
                                      bias,
                                      norm,
                                      allow_zero_in_degree)

        self._lambada_fun = SimpleMultiLayerNN(input_size=in_feats + out_feats,
                                               hidden_size=[floor(in_feats/2)],
                                               output_size=in_feats,
                                               out_act_fun=torch.nn.Sigmoid())

                                               #nn.Linear(in_feats + out_feats, in_feats)
        self._lambada_act = IdFun()