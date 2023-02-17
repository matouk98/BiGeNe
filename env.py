import torch.multiprocessing as mp
import time
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx

from player import Player
from utils.utils import entropy
from utils.config import parse_args
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


def percd(input):
    ranks = torch.argsort(input)
    ranks = (torch.argsort(ranks) + 1.0) / input.shape[0]
    return ranks

def logit2prob(logprobs,multilabel=False):
    if multilabel:
        probs = torch.sigmoid(logprobs)
    else:
        probs = F.softmax(logprobs, dim=-1)
    return probs

def normalizeEntropy(entro, classnum): #this is needed because different number of classes will have different entropy
    maxentro = np.log(float(classnum))
    entro = entro / maxentro  
    return entro

def perc(input):
    ranks = np.argsort(input, kind='stable')
    ranks = (np.argsort(ranks, kind='stable') + 1) / input.shape[0]
    return ranks

def centralissimo(G):
    centralities = nx.pagerank(G) #centralities.append(nx.harmonic_centrality(G))
    L = len(centralities)
    cenarray = np.zeros(L).astype(float)
    
    cenarray[list(centralities.keys())] = list(centralities.values())
    maxi, mini = np.max(cenarray), np.min(cenarray)
    normcen = (cenarray - mini) / (maxi-mini)
    
    return normcen

def localdiversity(probs, adj, deg):
    
    indices = adj.coalesce().indices()
    
    N = adj.size()[0]
    classnum = probs.size()[-1]
    maxentro = np.log(float(classnum))
    edgeprobs = probs[:,indices.transpose(0,1),:]
    
    headprobs = edgeprobs[:,:,0,:]
    tailprobs = edgeprobs[:,:,1,:]
    kl_ht = (torch.sum(torch.log(torch.clamp_min(tailprobs,1e-10))*tailprobs,dim=-1) - \
        torch.sum(torch.log(torch.clamp_min(headprobs,1e-10))*tailprobs,dim=-1)).transpose(0,1)
    kl_th = (torch.sum(torch.log(torch.clamp_min(headprobs,1e-10))*headprobs,dim=-1) - \
        torch.sum(torch.log(torch.clamp_min(tailprobs,1e-10))*headprobs,dim=-1)).transpose(0,1)

    sparse_output_kl_ht = torch.sparse.FloatTensor(indices,kl_ht,size=torch.Size([N,N,kl_ht.size(-1)]))
    sparse_output_kl_th = torch.sparse.FloatTensor(indices,kl_th,size=torch.Size([N,N,kl_th.size(-1)]))
    
    sum_kl_ht = torch.sparse.sum(sparse_output_kl_ht,dim=1).to_dense().transpose(0,1)
    sum_kl_th = torch.sparse.sum(sparse_output_kl_th,dim=1).to_dense().transpose(0,1)
    mean_kl_ht = sum_kl_ht/(deg+1e-10)
    mean_kl_th = sum_kl_th/(deg+1e-10)
    # normalize
    mean_kl_ht = mean_kl_ht / mean_kl_ht.max(dim=1, keepdim=True).values
    mean_kl_th = mean_kl_th / mean_kl_th.max(dim=1, keepdim=True).values
    return mean_kl_ht,mean_kl_th

def Euclidean_Distance(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

class Env(object):
    ## an environment for multiple players testing the policy at the same time
    def __init__(self, player, tgt_player, args):
        '''
        players: a list containing main player (many task) (or only one task
        '''
        self.player = player
        self.tgt_player = tgt_player
        self.args = args
        self.nplayer = 1
        self.graphs = [player.G, tgt_player.G]
        featdim = -1

        self.normadj = player.G.normadj
        self.tgt_normadj = tgt_player.G.normadj
        
        normcen = centralissimo(player.G.G)
        self.cenperc = torch.from_numpy(perc(normcen))
        self.statedim = self.getState().size(featdim)

        normcen = centralissimo(tgt_player.G.G)
        self.tgt_cenperc = torch.from_numpy(perc(normcen))
        self.tgt_statedim = self.getState('target')

        print("State Dim:{}".format(self.statedim))


    def step(self, actions, n_epochs=None, dataset="source"):
        if dataset == "target":
            p = self.tgt_player
        else:
            p = self.player
        p.query(actions)
        if n_epochs is None:
            n_epochs = self.args.train_epochs
        
        p.trainmodel(epochs=n_epochs)
        performance = p.validation(test=True, rerun=True)
        return performance

    def getState(self, dataset='source'):
        if dataset == "target":
            p = self.tgt_player
        else:
            p = self.player
        p.get_output()
        output = F.softmax(p.allnodes_output, dim=1)

        embeds_pair = None
        if self.args.state == 's4':
            embeds = self.get_embed(dataset)
            mask = p.trainmask.detach()
            has_select = torch.where(mask==1)[0]
            if not has_select.shape[0]:
                has_select = [1]
            
            y_embeds = embeds[has_select, :].detach()
            
            embeds_pair = (embeds, y_embeds)
        
        state = self.makeState(output, p.trainmask, multilabel=p.G.stat["multilabel"], dataset=dataset, embeds_pair=embeds_pair)
        
        return state

    def get_embed(self, dataset='source'):
        if dataset == "target":
            p = self.tgt_player
        else:
            p = self.player
        p.get_embed()
        return p.embeds

    def reset(self):
        self.player.reset()
        self.tgt_player.reset()
        self.performance = 0.0

    def uniform_actions(self, bs, dataset="source"):
        if dataset == "target":
            p = self.tgt_player
        else:
            p = self.player
        pool = p.getPool()
        n = pool.shape[0]
        sample_prob = torch.ones(n) / n
        random_sample = torch.multinomial(sample_prob, num_samples=bs)
        actions = pool[random_sample]
        
        return actions

    def greedy_actions(self, values, bs):
        p = self.player
        pool = p.getPool()

        in_pool = values[pool]
        in_rank = torch.argsort(in_pool, dim=0, descending=True)[:bs, :].view(-1)
        selected = pool[in_rank]
        
        '''
        pool_list = list(pool.numpy())
        maxi_score = -10000.0
        n = values.shape[0]
        for i in range(n):
            if values[i, 0] > maxi_score and i in pool_list:
                maxi_score = values[i, 0].item()
                select = i
        selected = selected.item()
        print(values[selected, :], values[select, :])
        print(selected, select)
        '''

        return selected, values[selected, :]
    
    def makeState(self, probs, selected, multilabel=False, dataset='target', embeds_pair=None):
        entro = entropy(probs, multilabel=multilabel)
        entro = normalizeEntropy(entro, probs.size(-1)) ## in order to transfer
        
        # entropy
        features = []
        if self.args.state >= 's2':
            features.append(entro.float().cuda())
        
        # centrality
        if self.args.state != 's2':
            if dataset == 'target':
                features.append(self.tgt_cenperc.float().cuda())
            else:
                features.append(self.cenperc.float().cuda())

        # node_embedding
        if embeds_pair is not None:
            x_embeds, y_embeds = embeds_pair
            
            coreset_dist = Euclidean_Distance(x_embeds, y_embeds)
            coreset_dist = torch.min(coreset_dist, dim=1)[0]
            coreset_dist /= torch.max(coreset_dist)
            features.append(coreset_dist)
            
        features.append(selected.float().cuda())
        state = torch.stack(features, dim=-1)

        return state

if __name__ == "__main__":
    from new_dataloader import GraphLoader
    
    args = parse_args()
    G = GraphLoader(args.dataset)
    
    p = Player(G, args)
    env = Env(p, p, args)
    env.reset()
    
    '''
    for i in range(args.query_cs):
        actions = env.uniform_actions(args.query_bs)
        reward = env.step(actions)
        print(reward)
    '''

    
