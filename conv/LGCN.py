import os
import sys
import dgl
import torch
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv
from torch import nn

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))


class LGCore(nn.Module):
    def __init__(self, in_feats, out_feats, agg_type, dropout, activation, update):
        super(LGCore, self).__init__()
        self.out_feats = out_feats
        self.agg_type = agg_type
        self.activation = activation
        self.update = update
        self.convLayer = GraphConv(in_feats, out_feats)
        self.fusionLayer = GraphConv(in_feats, out_feats)
        self.conv_w = nn.Parameter(torch.rand(out_feats))
        self.topDown_w = nn.Parameter(torch.rand(out_feats))
        self.bottomUp_w = nn.Parameter(torch.rand(out_feats))
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(out_feats)

    def forward(self, g, feats, inc):
        g = dgl.add_self_loop(g)
        if self.agg_type == 'top_down':
            curr_h = feats[0]
            next_h = feats[1]
            curr_inc = inc[0]

            conv_layer = self.convLayer(g, curr_h) * self.conv_w.unsqueeze(0)
            # conv_layer = self.batch_norm(conv_layer)
            # conv_layer = F.relu(conv_layer)

            top_down_layer = self.fusionLayer(g, torch.mm(curr_inc, next_h)) * self.topDown_w.unsqueeze(0)
            # top_down_layer = self.batch_norm(top_down_layer)
            # top_down_layer = F.relu(top_down_layer)

            if self.update == 'MEAN':
                result = torch.mean(torch.stack([conv_layer, top_down_layer]), dim=0)

            if self.update == 'MAX':
                result, _ = torch.max(torch.stack([conv_layer, top_down_layer]), dim=0)

            if self.update == 'SUM':
                result = conv_layer + top_down_layer

        elif self.agg_type == 'bottom_up':
            curr_h = feats[0]
            prev_h = feats[1]
            prev_inc_y = inc[0]

            conv_layer = self.convLayer(g, curr_h) * self.conv_w.unsqueeze(0)
            # conv_layer = self.batch_norm(conv_layer)
            # conv_layer = F.relu(conv_layer)

            bottom_up_layer = self.fusionLayer(g, torch.mm(prev_inc_y, prev_h)) * self.bottomUp_w.unsqueeze(0)
            # bottom_up_layer = self.batch_norm(bottom_up_layer)
            # bottom_up_layer = F.relu(bottom_up_layer)

            if self.update == 'MEAN':
                result = torch.mean(torch.stack([conv_layer, bottom_up_layer]), dim=0)

            if self.update == 'MAX':
                result, _ = torch.max(torch.stack([conv_layer, bottom_up_layer]), dim=0)

            if self.update == 'SUM':
                result = conv_layer + bottom_up_layer

        elif self.agg_type == 'both':
            curr_h = feats[0]
            prev_h = feats[1]
            next_h = feats[2]
            curr_inc = inc[0]
            prev_inc_y = inc[1]

            conv_layer = self.convLayer(g, curr_h) * self.conv_w.unsqueeze(0)
            # conv_layer = self.batch_norm(conv_layer)
            # conv_layer = F.relu(conv_layer)

            bottom_up_layer = self.fusionLayer(g, torch.mm(prev_inc_y, prev_h)) * self.bottomUp_w.unsqueeze(0)
            # bottom_up_layer = self.batch_norm(bottom_up_layer)
            # bottom_up_layer = F.relu(bottom_up_layer)

            top_down_layer = self.fusionLayer(g, torch.mm(curr_inc, next_h)) * self.topDown_w.unsqueeze(0)
            # top_down_layer = self.batch_norm(top_down_layer)
            # top_down_layer = F.relu(top_down_layer)

            if self.update == 'MEAN':
                result = torch.mean(torch.stack([conv_layer, bottom_up_layer, top_down_layer]), dim=0)

            if self.update == 'MAX':
                result, _ = torch.max(torch.stack([conv_layer, bottom_up_layer, top_down_layer]), dim=0)[0]

            if self.update == 'SUM':
                result = conv_layer + bottom_up_layer + top_down_layer

        # n = self.out_feats // 2
        # esult = self.layer_norm(result)
        # result = torch.cat([result[:, :n], F.relu(result[:, n:])], 1)
        result = self.layer_norm(result)

        if self.activation == 'RELU':
            result = F.relu(result)
        else:
            result = F.hardswish(result)

        result = self.dropout(result)

        return result


def get_inc(g, inc_type):
    return g.inc(inc_type)


class LGCN(nn.Module):
    def __init__(self, in_feats, out_feats, lg_list, inc_type, dropout, activation, update):
        super(LGCN, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.inc_type = inc_type
        self.lg_list = lg_list
        self.top_down = LGCore(self.in_feats, self.out_feats, 'top_down', dropout, activation, update)
        self.bottom_up = LGCore(self.in_feats, self.out_feats, 'bottom_up', dropout, activation, update)
        self.both = LGCore(self.in_feats, self.out_feats, 'both', dropout, activation, update)

    def forward(self, lg_h):
        new_lg_h = []
        for i in range(0, len(self.lg_list)):
            g = self.lg_list[i]
            current_h = lg_h[i]
            # finding the previous graph's hidden embedding
            # if it's the first graph, do only top-down
            if i == 0:
                # get the next graph's embedding
                next_h = lg_h[i + 1]
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
                prev_h = lg_h[i - 1]
                # get the prev graph's incident matrix
                prev_g = self.lg_list[i - 1]
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
                inc = [g_inc, prev_g_inc_y]
                # apply aggregation layer
                new_h = self.both(g, feats, inc)
                new_lg_h.append(new_h)

        return new_lg_h
