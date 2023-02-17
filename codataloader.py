import sys
import os
import torch
import numpy as np
import networkx as nx
from scipy.sparse import load_npz
from collections import OrderedDict
from networkx.convert_matrix import from_scipy_sparse_matrix
import scipy.sparse as sp

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_add_diag= adj + sp.eye(adj.shape[0])
    adj_normalized = normalize_adj(adj_add_diag)
    return adj_normalized.astype(np.float32) #sp.coo_matrix(adj_unnorm)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class CoauthorGraphLoader(object):

    def __init__(self, name, root = "./data"):
        self.name = name
        self.dirname = os.path.join(root,name)
       
        self.load()
        self.registerStat()
        self.printStat()
        self.process()

    def loadGraph(self):
        adj_path = os.path.join(self.dirname, 'adj.npz')
        self.adj = load_npz(adj_path)
        self.G = from_scipy_sparse_matrix(self.adj)
        self.normadj = preprocess_adj(self.adj)

        self.adj = sparse_mx_to_torch_sparse_tensor(self.adj)
        self.normadj = sparse_mx_to_torch_sparse_tensor(self.normadj)

    def loadX(self):
        feat_path = os.path.join(self.dirname, 'feat.npy')
        self.X = np.load(feat_path)
        
    def loadY(self):
        label_path = os.path.join(self.dirname, 'label.npy')
        self.Y = np.load(label_path)

    def load(self):
        self.loadGraph()
        self.loadX()
        self.loadY()

    def registerStat(self):
        L = OrderedDict()
        L["name"] = self.name
        L["nnode"] = self.G.number_of_nodes()
        L["nedge"] = self.G.number_of_edges()
        L["nfeat"] = self.X.shape[1]
        L["nclass"] = self.Y.max() + 1
        L['multilabel'] = False
        self.stat = L

    def process(self):
        # if int(self.bestconfig['feature_normalize']):
        #     self.X = column_normalize(preprocess_features(self.X)) # take some time
        
        self.X = torch.from_numpy(self.X).cuda()
        self.Y = torch.from_numpy(self.Y).cuda()
        self.adj = self.adj.cuda()
        self.normadj = self.normadj.cuda()
        
        self.normdeg = self.getNormDeg()
        self.get_test_index()

    def get_test_index(self):
        if os.path.exists(os.path.join(self.dirname, 'val.npy')):
            self.idx_val = np.load(os.path.join(self.dirname, 'val.npy')).tolist()
            self.idx_test = np.load(os.path.join(self.dirname, 'test.npy')).tolist()
        else:
            ntest = int(self.stat['nnode'] * 0.2)
            base = np.array([x for x in range(self.stat["nnode"])])
            np.random.seed(2018)
            self.idx_test = np.sort(np.random.choice(base, size=ntest, replace=False)).tolist()
        
    def printStat(self):
        print(self.stat)

    def getNormDeg(self):
        self.deg = torch.sparse.sum(self.adj, dim=1).to_dense()
        normdeg = self.deg / self.deg.max()
        return normdeg


if __name__ == "__main__":
    G = CoauthorGraphLoader('citeseer')
