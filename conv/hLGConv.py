import torch as th
import torch
from dgl import function as fn
from dgl.base import DGLError
from utils.utils_method import SimpleMultiLayerNN
from math import floor
from conv.LGConv import LGConv

class hLGConv(LGConv):
    def __init__(self,
                 in_feats,
                 out_feats,
                 k=1,
                 cached=False,
                 bias=True,
                 norm=None,
                 allow_zero_in_degree=False):
        super(hLGConv,self).__init__(in_feats,
                                      out_feats,
                                      k,
                                      cached,
                                      bias,
                                      norm,
                                      allow_zero_in_degree)

        self._lambada_fun = torch.nn.ModuleList()
        for i in range(self._k+1):
            self._lambada_fun.append(SimpleMultiLayerNN(in_feats, [floor(in_feats / 2)], 1))
        self._lambada_act = torch.nn.Sigmoid()



    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute Simplifying Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

        Note
        ----
        If ``cache`` is set to True, ``feat`` and ``graph`` should not change during
        training, or you will get wrong results.
        """
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
                     f=self._lambada_act(self._lambada_fun[i](k_feat))
                     result += self.fc(k_feat * self._alpha[i])* f
            else:
                feat_list = []

                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)
                # compute (D^-1 A^k D)^k X

                feat_list.append(feat.float())

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
                for i, k_feat in enumerate(feat_list):
                    f = self._lambada_act(self._lambada_fun[i](k_feat))
                    result += self.fc(k_feat * self._alpha[i]) * f

                if self.norm is not None:
                    result = self.norm(result)

                # cache feature
                if self._cached:
                    self._cached_h = feat_list
            return result
