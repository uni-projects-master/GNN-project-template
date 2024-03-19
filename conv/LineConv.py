import sys
path='C:/Users/solma/OneDrive/Documents/GitHub/Empowering-Simple-Graph-Convolutional-Networks'
sys.path.append(path)

import torch as th
from torch import nn
import torch
import numpy as np
from dgl import function as fn
from dgl.base import DGLError

from dgl.nn import GraphConv
from dgl.nn.pytorch.conv import GatedGraphConv


class LGCore(nn.Module):
    def __init__(self, in_feats, out_feats, w=0.7):
        super(LGCore, self).__init__()
        self.out_feats = out_feats
        self.conv = GraphConv(in_feats, out_feats)
        self.conv_fuse = GraphConv(in_feats, out_feats)
        self.layer_norm = nn.LayerNorm(out_feats)
        self.conv_w = nn.Parameter(torch.tensor(w))
        self.fusion_w = nn.Parameter(torch.tensor(1-w))

    def forward(self, g, feat_a, feat_b, inc):
        g = dgl.add_self_loop(g)
        conv_layer = self.conv(g, feat_a)
        fusion_layer = self.conv_fuse(g, torch.mm(inc, feat_b))
        # sum them together

        #result = conv_layer + fuse
        result = torch.add(torch.mul(conv_layer, self.conv_w), torch.mul(fusion_layer, self.fusion_w))
        
                
        # Skip Connections
        #n = self.out_feats // 2
        #result = torch.cat([result[:, :n], F.relu(result[:, n:])], 1)
        # Dropout in the model definition
        
        # Dropout Layer
        #result = self.dropout(result)
        
        # Layer Normalisation
        result = self.layer_norm(result)
        
        result = F.relu(result)
         
        return result


class LineConv(nn.Module):
    def __init__(self,
                 in_feats, 
                 out_feats.
                 k=1
                 cashed=False,
                 bias=True,
                 norm=None,
                 allow_zero_in_degree=False):
        super(LineConv, self).__init__()
        
        self.in_feats = in_feats
        self.out_feats = out_feats
        
        self.g_layer = LGCore(self.in_feats, self.out_feats)
        self.bottom_up = LGCore(self.in_feats, self.out_feats)        
        self.top_down = LGCore(self.in_feats, self.out_feats)        
               
        self._alpha=nn.ParameterList()

    def forward(self, g, lg, x, lg_x):
        
        g_inc = self.get_inc(g)
        new_x = F.relu(self.g_layer(g, x, lg_x[0], g_inc))
        
        new_lg_x = []
        for i in range(0, len(lg)):
            lg_i = lg[i]
            lg_i_x = lg_x[i]
            if i == 0:
                prev_x = x
                inc = g_inc
            else:
                prev_x = lg_x[i-1]
                inc = self.get_inc(lg[i-1])
            
            inc_y = torch.transpose(inc, 0, 1)
            a = self.bottom_up(lg_i, lg_i_x, prev_x, inc_y)
            if i == (len(lg)-1):
                new_lg_x.append(F.relu(a))
            else:
                next_x = lg_x[i+1]
                next_inc = self.get_inc(lg[i])
                b = self.top_down(lg_i, lg_i_x, next_x, next_inc)
                res = a+b
                new_lg_x.append(F.relu(res))
        
        
        #next_x = self.g_layer(g, x, lg_x, lg2_x, inc, inc2) 
        #next_lg_x = self.lg_layer(lg, lg_x, x, lg2_x, inc_y, inc2, lg=True)        
        #next_lg2_x = self.lg2_layer(lg2, lg2_x, lg_x, x, inc2_y, inc)
        
        return new_x, new_lg_x
    
    def get_inc(self, g):
        return g.inc('both')
 