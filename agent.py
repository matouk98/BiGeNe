from __future__ import print_function

import os
import sys
import time
import numpy as np
import torch
import networkx as nx
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mqnet import MQNet
from mem import NstepReplayMemCell
from utils.config import parse_args
from player import Player
from env import Env
from utils.utils import Tee

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def get_pool(mask):
    remain_pos = torch.where(mask<0.1)[0].cpu()
    return remain_pos

def load_graph(dataset_name):
    if dataset_name == "reddit1401":
        G = graphloader(dataset_name)
        G.process()
    elif dataset_name == 'cocs' or dataset_name == 'cophy' or dataset_name == 'citeseer':
        G = CoauthorGraphLoader(dataset_name)
    else:
        G = new_graphloader(dataset_name)
    
    return G

class Agent(object):
    def __init__(self, args, env, G, tgt_G, model_save_path):
        self.args = args
        self.G = G
        self.tgt_G = tgt_G
        
        self.mem_pool = NstepReplayMemCell(memory_size=500000, balance_sample=False)
        self.env = env 
        self.model_save_path = model_save_path
               
        self.net = MQNet(G, env.statedim, args)
        self.old_net = MQNet(G, env.statedim, args)
        
        self.net = self.net.cuda()
        self.old_net = self.old_net.cuda()

        self.net.reset_parameters()
        self.old_net.reset_parameters()

        self.old_net.eval()

        self.budget = self.args.query_cs * self.args.query_bs
        
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_step = 100000
        self.burn_in = 10    
        self.step = 0        
        self.pos = 0
        self.best_eval = 0.0
        self.take_snapshot()

    def take_snapshot(self):
        self.old_net.load_state_dict(self.net.state_dict())

    def make_actions(self, greedy=False, dataset='source'):
        # [num_candidate,]
        if dataset == 'source':
            pool = self.env.player.getPool()
        else:
            pool = self.env.tgt_player.getPool()
        
        self.eps = self.eps_end + max(0., (self.eps_start - self.eps_end)
                * (self.eps_step - max(0., self.step)) / self.eps_step)

        state = self.env.getState(dataset)

        if random.random() < self.eps and not greedy:
            greedy_flag = False
        else:
            greedy_flag = True

        if dataset == 'source':
            input_adj = self.G.normadj
        else:
            input_adj = self.tgt_G.normadj

        actions, q_values = self.net(state, pool, input_adj, self.args.query_bs, greedy_flag)

        return actions, q_values

    def run_simulation(self):
        self.env.reset()
        
        initial_actions = self.env.uniform_actions(20)
        now_perf = self.env.step(initial_actions, n_epochs=50)

        for i in range(self.args.query_cs):
            action_t, value_t = self.make_actions(greedy=False)
            state_t = self.env.getState().clone()

            self.env.player.model_reset()
            perf_t = self.env.step(action_t, n_epochs=50)
            reward_t = torch.Tensor([perf_t - now_perf])

            now_perf = perf_t
            
            if i == self.args.query_cs - 1:
                is_terminal = True
                s_prime = None
            else:
                is_terminal = False
                s_prime = (self.env.getState().clone(), self.env.player.trainmask+self.env.player.testmask+self.env.player.valmask)

            self.mem_pool.add_list(state_t, action_t, reward_t, s_prime, is_terminal)
            
    def eval(self):
        res_list = []
        for cs in range(50):
            self.env.reset()
            acc_list = []
            initial_actions = self.env.uniform_actions(20, dataset='target')
            now_perf = self.env.step(initial_actions, n_epochs=100, dataset='target')
            
            for i in range(self.args.query_cs):
                action_t, value_t = self.make_actions(greedy=True, dataset='target')
                self.env.tgt_player.model_reset()
                now_perf = self.env.step(action_t, n_epochs=100, dataset='target')
                acc_list.append(now_perf)
            acc_list = torch.FloatTensor(acc_list).view(1, -1)
            res_list.append(acc_list)

        final_res = torch.cat(res_list, dim=0)
        final_res = torch.mean(final_res, dim=0)
        out_str = ''
        for i in range(self.args.query_cs):
            out_str += '{:.3f}'.format(final_res[i].item()) +'\t'
        print(out_str)
        
        eval_acc = torch.sum(final_res[:]).item()
        if eval_acc > self.best_eval:
            self.best_eval = eval_acc
            print("Best Result!!")
            # torch.save(self.net.state_dict(), self.model_save_path)

    def test(self, path):
        tf = time.time()
        res_list = []
        self.net.load_state_dict(torch.load(path))
        
        for cs in range(50):
            self.env.reset()
            acc_list = []
            initial_actions = self.env.uniform_actions(20, dataset='target')
            now_perf = self.env.step(initial_actions, n_epochs=100, dataset='target')
            
            for i in range(self.args.query_cs):
                action_t, value_t = self.make_actions(greedy=True)
                self.env.tgt_player.model_reset()
                now_perf = self.env.step(action_t, n_epochs=100, dataset='target')
                acc_list.append(now_perf)
            acc_list = torch.FloatTensor(acc_list).view(1, -1)
            res_list.append(acc_list)

        final_res = torch.cat(res_list, dim=0)
        std = torch.std(final_res, dim=0)
        final_res = torch.mean(final_res, dim=0)

        out_str, std_str = '', ''
        for i in range(self.args.query_cs):
            if self.args.query_cs >= 20:
                if (i+1) % 2 == 0:
                    out_str += '{:.3f}'.format(final_res[i].item()) +'\t'
                    std_str += '{:.3f}'.format(std[i].item()) +'\t'
            else:
                out_str += '{:.3f}'.format(final_res[i].item()) +'\t'
                std_str += '{:.3f}'.format(std[i].item()) +'\t'

        print(out_str)
        print(std_str)
        print("Time takes:{:.2f}".format(time.time() - tf))

    def train(self):
        self.eval()
        for p in range(self.burn_in):
            self.run_simulation()
            
        optimizer = optim.Adam(self.net.parameters(), lr=self.args.rllr)
        tf = time.time()

        for self.step in range(self.args.rl_epoch):
            self.run_simulation()
            
            if (self.step+1) % self.args.rl_update == 0:
                self.take_snapshot()
            if (self.step+1) % 50 == 0:
                self.eval()

            for t in range(self.args.rl_train_iters):
                list_states, list_actions, list_rewards, list_s_primes, list_term = self.mem_pool.sample(batch_size=self.args.rl_bs)
                q_sa_list, q_target_list, q_real_list = [], [], []
                for i in range(self.args.rl_bs):
                    # [candidate, 3], [bs], [1], (next_state, mask), bool
                    
                    s, a, r, s_prime, is_terminal = list_states[i], list_actions[i], list_rewards[i], list_s_primes[i], list_term[i]
                    predicted_q_value = self.net.get_q_values(s, a, self.G.normadj)
                    q_sa = torch.mean(predicted_q_value).view(-1)
                    q_target = r.cuda()
                    q_real_list.append(r)

                    if not is_terminal:
                        pool = get_pool(s_prime[1])
                        action_t_plus_online, q_t_plus_online = self.net(s_prime[0], pool, self.G.normadj, greedy=True)
                        action_t_plus_online = action_t_plus_online.detach()

                        q_t_plus_target = self.old_net.get_q_values(s_prime[0], action_t_plus_online, self.G.normadj).detach()
                        q_maxi_t_plus = torch.mean(q_t_plus_target).view(-1)
                        # action_t_plus_target, q_t_plus_target = self.old_net(s_prime[0], pool, self.G.normadj, greedy=True)
                        # q_maxi_t_plus = torch.mean(q_t_plus_target).view(-1).cuda()

                        q_target += self.args.gamma * q_maxi_t_plus
                    
                    q_sa_list.append(q_sa)
                    q_target_list.append(q_target)
                    
                q_sa = torch.stack(q_sa_list)
                q_target = torch.stack(q_target_list)
                q_real = torch.stack(q_real_list)

                loss = F.smooth_l1_loss(q_sa, q_target)
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()

            if (self.step+1) % 20 == 0:
                print("Time:{:.2f}\tIteration:{}\tEps:{:.5f}\tLoss:{:.5f}\tQ_real:{:.5f}\tQ_target:{:.5f}".format(time.time()-tf, self.step, self.eps, loss.item(), torch.mean(q_real).item(), torch.mean(q_target).item()))
                tf = time.time()

            if (self.step+1) % 50 == 0:
                torch.save(self.net.state_dict(), self.model_save_path + '_{}.model'.format(self.step+1))

if __name__ == '__main__':
    from new_dataloader import GraphLoader as new_graphloader
    from utils.dataloader import GraphLoader as graphloader
    from codataloader import CoauthorGraphLoader
    args = parse_args()
    
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    log_file = "./logs/{}_{}_{}.txt".format(args.dataset, args.tgt_dataset, args.rl_update)
    Tee(log_file, "w")

    model_save_path = "./models/{}_{}_{}".format(args.dataset, args.tgt_dataset, args.rl_update)

    G = load_graph(args.dataset)
    tgt_G = load_graph(args.tgt_dataset)
    
    p = Player(G, args)
    tgt_p = Player(tgt_G, args)

    env = Env(p, tgt_p, args)
    agent = Agent(args, env, G, tgt_G, model_save_path)
    
    # agent.run_simulation()
    # agent.eval()
    agent.train()



    
