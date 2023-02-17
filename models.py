import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        if self.bias is not None:
            support = support + self.bias
        output = torch.spmm(adj, support)
        
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, softmax=True, bias=False):
        super(GCN, self).__init__()
        self.dropout_rate = dropout
        self.softmax = softmax
        
        self.gc1 = GraphConvolution(nfeat, nhid, bias=bias)
        self.gc2 = GraphConvolution(nhid, nclass, bias=bias)
        self.dropout = nn.Dropout(p=dropout, inplace=False)

    def get_embed(self, x, adj):
        x = self.dropout(x)
        x = F.relu(self.gc1(x, adj))
        return x

    def forward(self, x, adj, get_embed=False):
        
        x = self.dropout(x)
        x = F.relu(self.gc1(x, adj))

        embeds = self.dropout(x)
        x = self.gc2(embeds, adj)
        
        if get_embed:
            return embeds
        else:
            return x
        
    def reset(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()