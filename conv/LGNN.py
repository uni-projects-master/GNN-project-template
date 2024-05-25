import os
import sys
import dgl
import torch
import torch.nn.functional as F
from torch import nn
from dgl.nn.pytorch.conv import GraphConv


sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))


class LGCore(nn.Module):
    def __init__(self, in_feats, out_feats, agg_type, dropout, activation):
        super(LGCore, self).__init__()
        self.out_feats = out_feats
        self.agg_type = agg_type
        self.convLayer = GraphConv(in_feats, out_feats)
        self.fusionLayer = GraphConv(in_feats, out_feats)
        self.cat = nn.Linear(out_feats*2, out_feats)
        self.conv_w = nn.Parameter(torch.rand(out_feats))
        self.topDown_w = nn.Parameter(torch.rand(out_feats))
        self.bottomUp_w = nn.Parameter(torch.rand(out_feats))
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(out_feats)
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.LeakyReLU()

    def forward(self, g, feats, inc):
        g = dgl.add_self_loop(g)
        if self.agg_type == 'top_down':
            curr_h = feats[0]
            next_h = feats[1]
            curr_inc = inc[0]
            conv_layer_skip = self.convLayer(g, curr_h) * self.conv_w.unsqueeze(0)
            conv_layer_act = self.activation(conv_layer_skip)
            conv_layer = torch.cat((conv_layer_act, conv_layer_skip), dim=1)

            top_down_layer_skip = self.fusionLayer(g, torch.mm(curr_inc, next_h)) * self.topDown_w.unsqueeze(0)
            top_down_layer_act = self.activation(top_down_layer_skip)
            top_down_layer = torch.cat((top_down_layer_act, top_down_layer_skip), dim=1)

            result = torch.add(conv_layer, top_down_layer)

        elif self.agg_type == 'bottom_up':
            curr_h = feats[0]
            prev_h = feats[1]
            prev_inc_y = inc[0]
            conv_layer_skip = self.convLayer(g, curr_h) * self.conv_w.unsqueeze(0)
            conv_layer_act = self.activation(conv_layer_skip)
            conv_layer = torch.cat((conv_layer_act, conv_layer_skip), dim=1)

            bottom_up_layer_skip = self.fusionLayer(g, torch.mm(prev_inc_y, prev_h)) * self.bottomUp_w.unsqueeze(0)
            bottom_up_layer_act = self.activation(bottom_up_layer_skip)
            bottom_up_layer = torch.cat((bottom_up_layer_act, bottom_up_layer_skip), dim=1)

            result = torch.add(conv_layer, bottom_up_layer)

        elif self.agg_type == 'both':
            curr_h = feats[0]
            prev_h = feats[1]
            next_h = feats[2]
            prev_inc_y = inc[0]
            curr_inc = inc[1]
            conv_layer_skip = self.convLayer(g, curr_h) * self.conv_w.unsqueeze(0)
            conv_layer_act = self.activation(conv_layer_skip)
            conv_layer = torch.cat((conv_layer_act, conv_layer_skip), dim=1)

            bottom_up_layer_skip = self.fusionLayer(g, torch.mm(prev_inc_y, prev_h)) * self.bottomUp_w.unsqueeze(0)
            bottom_up_layer_act = self.activation(bottom_up_layer_skip)
            bottom_up_layer = torch.cat((bottom_up_layer_act, bottom_up_layer_skip), dim=1)

            top_down_layer_skip = self.fusionLayer(g, torch.mm(curr_inc, next_h)) * self.topDown_w.unsqueeze(0)
            top_down_layer_act = self.activation(top_down_layer_skip)
            top_down_layer = torch.cat((top_down_layer_act, top_down_layer_skip), dim=1)
            result = torch.add(conv_layer, bottom_up_layer)
            result = torch.add(result, top_down_layer)

        result = self.cat(result)
        result = self.layer_norm(result)
        result = self.dropout(result)

        return result


def get_inc(g, inc_type):
    return g.inc(inc_type)


class LGNN(nn.Module):
    def __init__(self, in_feats, out_feats, lg_list, inc_type, dropout, activation):
        super(LGNN, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.inc_type = inc_type
        self.lg_list = lg_list
        self.top_down = LGCore(self.in_feats, self.out_feats, 'top_down', dropout, activation)
        self.bottom_up = LGCore(self.in_feats, self.out_feats, 'bottom_up', dropout, activation)
        self.both = LGCore(self.in_feats, self.out_feats, 'both', dropout, activation)

    def forward(self, lg_h):
        new_lg_h = []
        for i in range(0, len(self.lg_list)):
            g = self.lg_list[i]
            current_h = lg_h[i]
            # finding the previous graph's hidden embedding
            # if it's the first graph, do only top-down
            if i == 0:
                # get the next graph's embedding
                next_h = lg_h[i+1]
                # get the incident matrix B_0
                g_inc = get_inc(g, self.inc_type)
                # apply top-down aggregation
                feats = [current_h, next_h]
                inc = [g_inc]
                new_h = self.top_down(g, feats, inc)
                new_lg_h.append(new_h)
            # if it's the last graph in the hierarchy, do only bottom-up
            elif i == (len(self.lg_list) - 1):
                # get the prev graph's embedding
                prev_h = lg_h[i-1]
                # get the prev graph's incident matrix
                prev_g = self.lg_list[i-1]
                prev_g_inc = get_inc(prev_g, self.inc_type)
                prev_g_inc_y = torch.transpose(prev_g_inc, 0, 1)
                feats = [current_h, prev_h]
                inc = [prev_g_inc_y]
                new_h = self.bottom_up(g, feats, inc)
                new_lg_h.append(new_h)
            # if it's in-between, apply both
            else:
                # get next and prev graphs' embeddings
                prev_h = lg_h[i - 1]
                next_h = lg_h[i + 1]
                feats = [current_h, prev_h, next_h]
                # get curr and prev graphs' inc matrix (B_i, B_(i-1)^T)
                g_inc = get_inc(g, self.inc_type)
                prev_g = self.lg_list[i - 1]
                prev_g_inc = get_inc(prev_g, self.inc_type)
                prev_g_inc_y = torch.transpose(prev_g_inc, 0, 1)
                inc = [prev_g_inc_y, g_inc]
                # apply aggregation layer
                new_h = self.both(g, feats, inc)
                new_lg_h.append(new_h)

        return new_lg_h
