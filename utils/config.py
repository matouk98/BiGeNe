import numpy as np
import torch
import time
import os
import argparse
from pprint import pformat
from torch.distributions import Categorical
import torch.nn.functional as F
from utils.log import * 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nhid",type=int,default=16)
    parser.add_argument("--pnhid",type=str,default='8+8')
    parser.add_argument("--dropout",type=float,default=0.5)
    parser.add_argument("--pdropout",type=float,default=0.0)
    parser.add_argument("--lr",type=float,default=1e-2)
    
    parser.add_argument("--rllr",type=float,default=1e-3)
    parser.add_argument("--rl_bs",type=int,default=16)
    parser.add_argument("--rl_epoch",type=int,default=350)
    parser.add_argument("--rl_update",type=int,default=20)
    parser.add_argument("--rl_train_epoch",type=int,default=50)
    parser.add_argument("--rl_train_iters",type=int,default=10)
    parser.add_argument("--gamma",type=float,default=0.99)
    parser.add_argument("--rl_hid",type=int,default=8)
    parser.add_argument("--use_dist",type=int,default=1)

    parser.add_argument("--batchsize",type=int,default=1)
    parser.add_argument("--ntest",type=int,default=1000)
    parser.add_argument("--nval",type=int,default=500)
    parser.add_argument("--dataset",type=str,default="cora")
    parser.add_argument("--tgt_dataset",type=str,default="cora")
    parser.add_argument("--state",type=str,default="s4")
    
    parser.add_argument("--query_bs",type=int,default=20)
    parser.add_argument("--query_cs",type=int,default=10)

    parser.add_argument("--logfreq",type=int,default=200)
    parser.add_argument("--maxepisode",type=int,default=20000)
   
    parser.add_argument("--policynet",type=str,default='gcn')
    parser.add_argument("--multigraphindex", type=int, default=1)

    parser.add_argument("--use_entropy",type=int,default=1)
    parser.add_argument("--use_degree",type=int,default=1)
    parser.add_argument("--use_local_diversity",type=int,default=1)
    parser.add_argument("--use_select",type=int,default=1)

    args = parser.parse_args()
    logargs(args,tablename="config")
    args.pnhid = [int(n) for n in args.pnhid.split('+')]

    return args
