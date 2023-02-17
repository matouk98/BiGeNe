import  numpy as np
import  pickle as pkl
import  networkx as nx
import  scipy.sparse as sp
from    scipy.sparse.linalg.eigen.arpack import eigsh
import  sys
import os
import torch

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
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_mx_to_torch_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch tensor."""
    return torch.FloatTensor(sparse_mx.astype(np.float32).toarray())


def parse_index_file(filename):
    """
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_graph(file_name):
    G = nx.Graph()
    L = open(file_name, "r").readlines()
    num_node = int(L[0].strip())
    L = L[1:]
    edge_list = [[int(x) for x in e.strip().split()] for e in L]

    G.add_nodes_from([x for x in range(num_node)])
    G.add_edges_from(edge_list)
    return G

class GraphLoader(object):
    def __init__(self, name, root="./data", args=None):
        self.name = name
        self.args = args
        self.data_path = os.path.join(root, name, "raw")
        self.edge_path = os.path.join(root, name, "{}.edgelist".format(name))
        self.load_data(name)

    def load_data(self, dataset_str):
        """
        Loads input data from gcn/data directory
        ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
        ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
        All objects above must be saved using python pickle module.
        :param dataset_str: Dataset name
        :return: All data input files loaded (as well the training/test data).
        """
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        
        for i in range(len(names)):
            with open("{}/ind.{}.{}".format(self.data_path, dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(self.data_path, dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        # print(np.min(test_idx_range), np.max(test_idx_range))
        
        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        #self.adj = np.array(adj.todense())
        
        self.G = load_graph(self.edge_path)

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)

        labels = torch.from_numpy(labels)
        self.Y = torch.argmax(labels, dim=1)
        self.idx_test = idx_test

        adj_normalized = preprocess_adj(adj)
        adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
        features = torch.FloatTensor(features.todense())

        self.X = features.cuda()
        self.normadj = adj_normalized.cuda()

        self.stat = {}
        self.stat['name'] = dataset_str
        self.stat['multilabel'] = False
        self.stat['nnode'] = self.X.shape[0]
        self.stat['nfeat'] = self.X.shape[1]
        self.stat['nclass'] = y.shape[1]
        print(self.stat)

    def process(self):
        return


if __name__ == "__main__":
    G = GraphLoader("pubmed")
    
    