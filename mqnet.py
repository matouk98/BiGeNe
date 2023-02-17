from __future__ import print_function

import os
import sys
import numpy as np
import torch
import networkx as nx
import random
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from models import GraphConvolution
from utils.utils import uniform_actions_pool, greedy_actions_pool

class MQNet(nn.Module):
    def __init__(self, G, state_dim, args):
        super(MQNet, self).__init__()
        self.state_dim = state_dim
        self.args = args
        
        self.gcn_layer1 = GraphConvolution(state_dim, args.rl_hid, bias=True)
        self.gcn_layer2 = GraphConvolution(args.rl_hid, args.rl_hid, bias=True)
        self.output_layer1 = nn.Linear(args.rl_hid, 1)
        self.output_layer2 = nn.Linear(args.rl_hid, 1)
        self.rnn = nn.GRUCell(args.rl_hid, args.rl_hid)
        
    def gcn_forward(self, state, adj):
        feats = F.relu(self.gcn_layer1(state, adj))
        feats = F.relu(self.gcn_layer2(feats, adj))
    
        values = self.output_layer1(feats)
       
        return values, feats

    def rnn_forward(self, state, hidden_state):
        new_feats = self.rnn(state, hidden_state)
        values = self.output_layer2(new_feats)

        return values, new_feats

    def forward(self, state, pool, adj, bs=20, greedy=False):
        actions, q_values = [], []
        values, gcn_feats = self.gcn_forward(state, adj)
        # values, gcn_feats = values.detach(), gcn_feats.detach()
        n = gcn_feats.shape[0]
        if not greedy:
            pos, first_action, first_value = uniform_actions_pool(values, pool)
        else:
            pos, first_action, first_value = greedy_actions_pool(values, pool)

        actions.append(first_action)
        q_values.append(first_value)

        pos = pos.item()
        hidden_state = gcn_feats[first_action.item(), :].repeat(n, 1)
        pool = np.delete(pool, pos)

        for i in range(bs-1):
            rnn_feats = self.rnn(gcn_feats, hidden_state)
            values = self.output_layer2(rnn_feats)

            if not greedy:
                pos, action, value = uniform_actions_pool(values, pool)
            else:
                pos, action, value = greedy_actions_pool(values, pool)

            actions.append(action)
            q_values.append(value)

            pos = pos.item()
            hidden_state = rnn_feats[action.item(), :].repeat(n, 1)
            pool = np.delete(pool, pos)

        actions = torch.from_numpy(np.stack(actions))
        q_values = torch.cat(q_values).detach().cpu()
        
        return actions, q_values

    def get_q_values(self, state, action, adj, bs=20):
        q_values = []
        
        values, gcn_feats = self.gcn_forward(state, adj)
        n = gcn_feats.shape[0]
        first_action = action[0].item()
        q_values.append(values[first_action])

        hidden_state = gcn_feats[first_action, :].repeat(n, 1)
        for i in range(bs-1):
            rnn_feats = self.rnn(gcn_feats, hidden_state)
            values = self.output_layer2(rnn_feats)
            a = action[i+1].item()
            q_values.append(values[a])
            hidden_state = rnn_feats[a, :].repeat(n, 1)

        q_values = torch.stack(q_values)
        return q_values

    def reset_parameters(self):
        weights_init(self)

class PQNet(nn.Module):
    def __init__(self, G, state_dim, args):
        super(PQNet, self).__init__()
        self.node_features = G.X
        self.adj = G.normadj
        self.state_dim = state_dim
        self.args = args
        
        self.gcn_layer1 = GraphConvolution(state_dim, args.rl_hid, bias=True)
        self.gcn_layer2 = GraphConvolution(args.rl_hid, args.rl_hid, bias=True)
        self.value_fc1 = nn.Linear(args.rl_hid, 1)
        self.value_fc2 = nn.Linear(args.rl_hid, 1)
        self.rnn = nn.GRUCell(args.rl_hid, args.rl_hid)
        
    def gcn_forward(self, state, adj):
        feats = F.relu(self.gcn_layer1(state, adj))
        feats = F.relu(self.gcn_layer2(feats, adj))
    
        values = self.value_fc1(feats)
        return values, feats

    def forward(self, state, pool, adj, bs=20, greedy=False):
        actions, q_values = [], []
        values, gcn_feats = self.gcn_forward(state, adj)
        # values, gcn_feats = values.detach(), gcn_feats.detach()
        n = gcn_feats.shape[0]
        if not greedy:
            pos, first_action, first_value = uniform_actions_pool(values, pool)
        else:
            pos, first_action, first_value = greedy_actions_pool(values, pool)

        actions.append(first_action)
        q_values.append(first_value)

        pos = pos.item()
        hidden_state = gcn_feats[first_action.item(), :].view(1, -1)
        pool = np.delete(pool, pos)

        for i in range(bs-1):
            # new_feats = torch.cat((gcn_feats, hidden_state.repeat(n, 1)), dim=1)
            new_feats = (gcn_feats + hidden_state) / 2

            values = self.value_fc2(new_feats)
            # values = values.detach()
            
            if not greedy:
                pos, action, value = uniform_actions_pool(values, pool)
            else:
                pos, action, value = greedy_actions_pool(values, pool)

            actions.append(action)
            q_values.append(value)

            pos = pos.item()
            selected_state = gcn_feats[action.item(), :].view(1, -1)
            pool = np.delete(pool, pos)
            hidden_state = self.rnn(selected_state, hidden_state).view(1, -1)
            
        actions = torch.from_numpy(np.stack(actions))
        q_values = torch.cat(q_values).detach().cpu()
        
        return actions, q_values

    def get_q_values(self, state, action, adj, bs=20):
        q_values = []
        
        values, gcn_feats = self.gcn_forward(state, adj)
        n = gcn_feats.shape[0]
        first_action = action[0].item()
        q_values.append(values[first_action])

        hidden_state = gcn_feats[first_action, :].view(1, -1)
        for i in range(bs-1):
            new_feats = (gcn_feats + hidden_state) / 2
            # new_feats = torch.cat((gcn_feats, hidden_state.repeat(n, 1)), dim=1)
            values = self.value_fc2(new_feats)
            a = action[i+1].item()
            q_values.append(values[a])

            selected_state = gcn_feats[a, :].view(1, -1)
            hidden_state = self.rnn(selected_state, hidden_state).view(1, -1)
            
        q_values = torch.stack(q_values)
        return q_values

    def reset_parameters(self):
        weights_init(self)
        
def glorot_uniform(t):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        # out_ch, in_ch, kernel for Conv 1
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)


def _param_init(m):
    if isinstance(m, Parameter):
        glorot_uniform(m.data)
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
        glorot_uniform(m.weight.data)

def weights_init(m):
    for p in m.modules():
        if isinstance(p, nn.ParameterList):
            for pp in p:
                _param_init(pp)
        else:
            _param_init(p)

    for name, p in m.named_parameters():
        if not '.' in name: # top-level parameters
            _param_init(p)