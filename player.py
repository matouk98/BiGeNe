# individual player who takes the action and evaluates the effect
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import numpy as np
import sys

sys.path.append('..')
sys.path.append('.')

from utils.utils import accuracy
from models import GCN
from utils.config import parse_args
from copy import deepcopy

class Player(nn.Module):

    def __init__(self,G, args, rank=0):

        super(Player,self).__init__()
        self.G = G
        self.args = args
        self.rank = rank
        
        if self.G.stat['multilabel']:
            self.net = GCN(self.G.stat['nfeat'], args.nhid, self.G.stat['nclass'], args.dropout,False,bias=True).cuda()
            self.loss_func = F.binary_cross_entropy_with_logits
        else:
            self.net = GCN(self.G.stat['nfeat'], args.nhid, self.G.stat['nclass'], args.dropout,True).cuda()
            self.loss_func = nn.CrossEntropyLoss(reduction='none')

        self.fulllabel = self.G.Y.cuda()
        self.reset() #initialize
        self.count = 0

    def makeValTestMask(self, fix_test=False):
        valmask = torch.zeros(self.G.stat['nnode']).to(torch.float).cuda()
        testmask = torch.zeros(self.G.stat['nnode']).to(torch.float).cuda()
       
        base = np.array([x for x in range(self.G.stat["nnode"])])

        if hasattr(self.G, 'idx_test'):
            testid = self.G.idx_test
        else:
            if fix_test:
                testid =[x for x in range(self.G.stat["nnode"] - self.args.ntest, self.G.stat["nnode"])]
            else:
                testid = np.sort(np.random.choice(base, size=self.args.ntest, replace=False)).tolist()
        testmask[testid] = 1.
        self.testlabel = self.G.Y[testid]
        
        if hasattr(self.G, 'idx_val'):
            valid = self.G.idx_val
        else:
            s = set(testid)
            base = [x for x in range(self.G.stat["nnode"]) if x not in s]
            valid = np.sort(np.random.choice(base, size=self.args.nval, replace=False)).tolist()
        valmask[valid]=1.
        self.vallabel = self.G.Y[valid]
        
        self.valid = torch.tensor(valid).cuda()
        self.testid = torch.tensor(testid).cuda()
        self.vallabel = self.vallabel.cuda()
        self.testlabel = self.testlabel.cuda()
        self.valmask = valmask
        self.testmask = testmask

    def query(self, nodes):
        list_nodes = list(nodes.cpu().numpy())
        for node in list_nodes:
            if self.trainmask[node] + self.valmask[node] + self.testmask[node] > 0:
                print("Error")
        
        self.trainmask[nodes] = 1.
    
    def getPool(self, reduce=True):
        mask = self.testmask + self.valmask + self.trainmask
        
        remain_pos = torch.where(mask<0.1)[0].cpu()
        return remain_pos

    def trainOnce(self,log=False):
        nlabeled = torch.sum(self.trainmask)
        self.net.train()
        self.opt.zero_grad()
        output = self.net(self.G.X, self.G.normadj)
        
        losses = self.loss_func(output, self.fulllabel)
        
        loss = torch.sum(losses * self.trainmask) / nlabeled
        loss.backward()
        self.opt.step()
        
        return output

    def get_output(self):
        self.net.eval()
        self.allnodes_output = self.net(self.G.X, self.G.normadj).detach()

    def get_embed(self):
        self.net.eval()
        self.embeds = self.net.get_embed(self.G.X, self.G.normadj).detach()
        
    def validation(self, test=False, rerun=True):
        if test:
            mask = self.testmask
            labels = self.testlabel
            index = self.testid
        else:
            mask = self.valmask
            labels = self.vallabel
            index = self.valid
        if rerun:
            self.net.eval()
            output = self.net(self.G.X, self.G.normadj)
        else:
            output = self.allnodes_output
        
        pred_val = output[index, :]
        return torch.Tensor([accuracy(pred_val, labels)])

    def trainmodel(self, epochs):
        best_val, best_test = 0.0, 0.0
        val_list = []
        for i in range(epochs):
            self.trainOnce()
            val_acc = self.validation()
            test_acc = self.validation(test=True)
            if val_acc > best_val:
                best_test = test_acc
                best_val = val_acc
                best_model = deepcopy(self.net)
        
        self.net.load_state_dict(best_model.state_dict())

    def model_reset(self):
        self.net.reset()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=5e-4)

    def reset(self,resplit=True):
        if resplit:
            self.makeValTestMask()
        self.trainmask = torch.zeros(self.G.stat['nnode']).to(torch.float).cuda()
        self.net.reset()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=5e-4)
        self.allnodes_output = self.net(self.G.X, self.G.normadj).detach()
       

if __name__=="__main__":
    from utils.dataloader import GraphLoader
    args = parse_args()
    G = GraphLoader("cora")
    G.process()
    p = Player(G, args)
    pool = p.getPool()
    
    
    n = pool.shape[0]
    sample_prob = torch.ones(n) / n
    random_sample = torch.multinomial(sample_prob, num_samples=140)
    actions = pool[random_sample]
    p.query(actions)
   
    p.trainmodel(100)
    print(p.validation(test=True))
    
    