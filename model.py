from dgc.utils import load_graph_data, normalize_adj, construct_filter, normalize_adj_torch
from dgc.clustering import k_means
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from torch_geometric.nn import DMoNPooling, GCNConv

from torch_geometric.nn.models.mlp import MLP
from typing import Optional, Tuple
from torch_geometric.nn import Linear
from math import log
from torch.nn import ModuleList
from torch import Tensor

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import MLP


from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch.nn import Parameter
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_layers=2):
        super().__init__()
        self.fc = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(n_layers-1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.last_conv = GCNConv(hidden_channels, hidden_channels)
        

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x).relu()
        for conv in self.convs:
            x = F.dropout(x, p=0.5, training=self.training)
            x = conv(x, edge_index, edge_weight).relu()

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.last_conv(x, edge_index, edge_weight)
        return x


class encoding(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, low_pass_filter):
        super(encoding, self).__init__()
        self.args = args

        if args.first_transformation=='mlp':
            self.first_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim))
        elif args.first_transformation=='linear':
            self.first_layer = nn.Linear(input_dim, hidden_dim)
        else:
            raise NotImplementedError
        
        self.low_pass_filter = low_pass_filter
        self.low_pass_fc  = nn.Linear(hidden_dim, hidden_dim)
        self.high_pass_fc = nn.Linear(hidden_dim, hidden_dim)

        if args.fusion_method == 'concat':
            self.fusion_fc = nn.Linear(2*hidden_dim, 2*hidden_dim)
            # self.fusion_param = nn.Parameter(torch.FloatTensor(2*hidden_dim, 2*hidden_dim))
            # self.last_layer = nn.Linear(2*hidden_dim, output_dim)
        elif args.fusion_method == 'add' or args.fusion_method == 'max':
            self.fusion_fc = nn.Linear(hidden_dim, hidden_dim)
            # self.fusion_param = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
            # self.last_layer = nn.Linear(hidden_dim, output_dim)
        else:
            raise NotImplementedError
    
        

    def construct_high_pass_adj(self, H_0, adj):
        H_0_norm = F.normalize(H_0, p=2, dim=1)
        pair_wise_cos = torch.mm(H_0_norm, H_0_norm.t())
        pair_wise_cos = pair_wise_cos * (adj!=0)
        return F.relu(pair_wise_cos)
    

    def fusion(self, H_low, H_high):
        if self.args.fusion_method == 'add':
            H =  self.args.fusion_beta * H_low + (1-self.args.fusion_beta) * H_high
        elif self.args.fusion_method == 'concat':
            H = torch.cat((self.args.fusion_beta*H_low, (1-self.args.fusion_beta)*H_high), dim=1)
        elif self.args.fusion_method == 'max':
            H = torch.max(H_low, H_high)
        else:
            raise NotImplementedError
        return H
    

    def forward(self, X, adj):
        H_0 = self.first_layer(X)
        self.high_pass_adj = normalize_adj_torch(self.construct_high_pass_adj(H_0, adj))
        self.high_pass_filter = construct_filter(adj=self.high_pass_adj, 
                                                 l=self.args.high_pass_layers, 
                                                 alpha=self.args.high_pass_alpha)

        self.H_low = self.low_pass_fc(self.low_pass_filter @ H_0)
        self.H_high = self.high_pass_fc(self.high_pass_filter @ H_0)
        H = self.fusion(self.H_low, self.H_high)        

        H = H + self.args.fusion_gamma * self.fusion_fc(H)
        # H = self.last_layer(H)
        # H = F.softmax(H, dim=1)
        
        return H


class encoding2(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, norm_adj, cluster_num):
        super(encoding2, self).__init__()
        self.args = args

        if args.first_transformation=='mlp':
            self.first_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim))
        elif args.first_transformation=='linear':
            self.first_layer = nn.Linear(input_dim, hidden_dim)
        else:
            raise NotImplementedError
        
        self.norm_adj = norm_adj
        self.cluster_num = cluster_num

        self.lins = ModuleList()
        for i in range(self.args.low_pass_layers):
            self.lins.append(Linear(in_channels = hidden_dim, out_channels = hidden_dim, bias = True, weight_initializer = 'glorot'))

        self.low_pass_fc  = nn.Linear(hidden_dim, hidden_dim)
        self.high_pass_fc = nn.Linear(hidden_dim, hidden_dim)

        self.final_layer = Linear(in_channels = hidden_dim, out_channels = hidden_dim, bias = True, weight_initializer = 'glorot')

        self.cluster_layer = Linear(in_channels = hidden_dim, out_channels = self.cluster_num, bias = True, weight_initializer = 'glorot')
        #nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                                             # nn.ReLU(),
                                             # nn.Linear(hidden_dim, self.cluster_num))

        if args.fusion_method == 'concat':
            self.fusion_fc = nn.Linear(2*hidden_dim, 2*hidden_dim)
            # self.fusion_param = nn.Parameter(torch.FloatTensor(2*hidden_dim, 2*hidden_dim))
            # self.last_layer = nn.Linear(2*hidden_dim, output_dim)
        elif args.fusion_method == 'add' or args.fusion_method == 'max':
            self.fusion_fc = nn.Linear(hidden_dim, hidden_dim)
            # self.fusion_param = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
            # self.last_layer = nn.Linear(hidden_dim, output_dim)
        else:
            raise NotImplementedError
    
        

    def construct_high_pass_adj(self, H_0):
        H_0_norm = F.normalize(H_0, p=2, dim=1)
        pair_wise_cos = torch.mm(H_0_norm, H_0_norm.t())
        pair_wise_cos = pair_wise_cos * (self.norm_adj!=0)
        return F.relu(pair_wise_cos)
    

    def fusion(self, H_low, H_high):
        if self.args.fusion_method == 'add':
            H =  self.args.fusion_beta * H_low + (1-self.args.fusion_beta) * H_high
        elif self.args.fusion_method == 'concat':
            H = torch.cat((self.args.fusion_beta*H_low, (1-self.args.fusion_beta)*H_high), dim=1)
        elif self.args.fusion_method == 'max':
            H = torch.max(H_low, H_high)
        else:
            raise NotImplementedError
        return H
    

    def forward(self, X, mask: Optional[Tensor] = None):
        H_0 = self.first_layer(X)
        # self.high_pass_adj = normalize_adj_torch(self.construct_high_pass_adj(H_0))
        # self.high_pass_filter = construct_filter(adj=self.high_pass_adj, 
                                                 # l=self.args.high_pass_layers, 
                                                 # alpha=self.args.high_pass_alpha)
        H = H_0
        for i in range(self.args.low_pass_layers):
            # temp = torch.mm(H_0.t(), H)
            # temp = torch.mm(H_0, temp)
            # DeProp(H, H_0, self.norm_adj, self.args.step_size_gamma, self.args.alphaH, self.args.alphaO)
            # H = F.normalize(H, p=2, dim=1)

            H = self.args.alphaH * torch.spmm(self.norm_adj, H) + H_0
            H = (1-self.args.alphaH) * torch.spmm(self.norm_adj, H) + self.args.alphaH * H_0
            H = DePropagate(H, H_0, self.norm_adj, self.args.step_size_gamma, self.args.alphaH, self.args.alphaO)
            # H = F.normalize(H, p=2, dim=1)



            # H_low = self.alpha*self.norm_adj(H_low) + H_0

            # theta = log(self.args.gamma / (i + 2))
            # H = (1 - theta) * H + theta * self.lins[i].forward(H)
            # H = H + self.lins[i].forward(H)
            # H = F.dropout(H, p = self.args.dropout, training = self.training, inplace = True)
            # H = F.relu(H, inplace = True)
            # H = F.selu(H, inplace = True)

        # self.H_high = self.high_pass_fc(self.high_pass_filter @ H_0)
        # H = self.fusion(self.H_low, self.H_high)        

        # H = H + self.args.fusion_gamma * self.fusion_fc(H)
        H = self.final_layer(H)
        # H = F.dropout(H, p = self.args.dropout, training = self.training, inplace = True)
        H = F.normalize(H, p=2, dim=1)

        
        return H



class top_agg(nn.Module):
    def __init__(self, A_norm, alpha, hop, emb_dim, hidden_dim, linear_prop='sgc', linear_trans='lin', norm=None):
        super(top_agg, self).__init__()

        if linear_prop=='sgc':
            I = torch.eye(A_norm.shape[0]).to(A_norm.device)
            top_filter = torch.eye(A_norm.shape[0]).to(A_norm.device)
            for _ in range(hop):
                top_filter = alpha * A_norm @ top_filter + I
            self.top_filter = top_filter
            if linear_trans=='lin':
                self.fc = nn.Linear(emb_dim, emb_dim)
            elif linear_trans=='mlp':
                self.fc = MLP(in_channels=emb_dim, hidden_channels=hidden_dim, out_channels=emb_dim, num_layers=2, batch_norm=False, dropout=0.0, bias=True)
        elif linear_prop=='gcn':
            self.top_filter = A_norm
            self.lins = ModuleList()
            self.lins.append(Linear(in_channels = emb_dim, out_channels = hidden_dim, bias = True, weight_initializer = 'glorot'))
            for i in range(hop-2):
                self.lins.append(Linear(in_channels = hidden_dim, out_channels = hidden_dim, bias = True, weight_initializer = 'glorot'))
            self.lins.append(Linear(in_channels = hidden_dim, out_channels = emb_dim, bias = True, weight_initializer = 'glorot'))
        else:
            raise NotImplementedError
        self.linear_prop = linear_prop
        self.norm = norm
        

    def agg(self, x):
        return self.top_filter @ x

    def forward(self, x):
        if self.linear_prop == 'sgc':
            x = self.fc(x)
            # x = (x - x.mean(0)) / x.std(0) / torch.sqrt(torch.tensor(x.shape[1]).to(x.device))
            if self.norm == 'l2-norm':
                print('l2-norm')
                x = F.normalize(x, p=2, dim=1)
        elif self.linear_prop == 'gcn':
            for lin in self.lins[:-1]:
                x = lin(self.top_filter @ x)
                x = F.relu(x)
            x = self.lins[-1](self.top_filter @ x)
        
        
        return x









class top_agg2(nn.Module):
    def __init__(self, A_norm, alpha, hop, emb_dim, hidden_dim, linear_trans='lin'):
        super(top_agg2, self).__init__()

        self._top_filter(A_norm, alpha, hop)

        if linear_trans=='lin':
            self.fc = nn.Linear(emb_dim, emb_dim)
        elif linear_trans=='mlp':
            self.fc = MLP(in_channels=emb_dim, hidden_channels=hidden_dim, out_channels=emb_dim, num_layers=2, batch_norm=False, dropout=0.0, bias=True)
    
    
    def _top_filter(self, A_norm, alpha, hop):
        I = torch.eye(A_norm.shape[0]).to(A_norm.device)
        top_filter = torch.eye(A_norm.shape[0]).to(A_norm.device)
        for _ in range(hop):
            top_filter = alpha * A_norm @ top_filter + I
        self.top_filter = top_filter

    def agg(self, x):
        return self.top_filter @ x

    def forward(self, x):
        x = self.fc(x)
        
        return x







def compute_attr_simi_mtx(X, attr_r, bin=0):
    ### contruct the attribute similarity matrix ###
    X_n = F.normalize(X, p=2, dim=1)
    attr_simi_mtx = (X_n@X_n.t()).to(X.device)
    attr_simi_mtx[attr_simi_mtx<0] = 0
    # attr_simi_mtx = attr_simi_mtx - torch.diag_embed(torch.diag(attr_simi_mtx))

    row, col = torch.nonzero(attr_simi_mtx, as_tuple=True)
    values = attr_simi_mtx[row, col] 
    _, sort_indices = torch.sort(values, descending=True)

    keep_size = int(attr_r * len(sort_indices))
    sort_indices = sort_indices[:keep_size]

    values = values[sort_indices]
    if bin == 1:
        values = torch.ones_like(values)
    row = row[sort_indices]
    col = col[sort_indices]
    attr_simi_mtx = torch.sparse_coo_tensor(torch.stack((row, col),0), values, attr_simi_mtx.shape)
    attr_simi_mtx = attr_simi_mtx.to_dense()

    return attr_simi_mtx


def compute_gaussian_kernel_mtx(X, sigma=0.2):
    X_n = F.normalize(X, p=2, dim=1)
    attr_simi_mtx = (X_n@X_n.t()).to(X.device)
    attr_simi_mtx = torch.exp(-attr_simi_mtx/(2 * sigma ** 2))
    return attr_simi_mtx


def compute_knn_simi_mtx(X, k=10):
    X_n = F.normalize(X, p=2, dim=1)
    attr_simi_mtx = (X_n@X_n.t()).to(X.device)
    _, indices = torch.topk(attr_simi_mtx, k, dim=1)
    row = torch.arange(attr_simi_mtx.shape[0]).repeat(k, 1).t().reshape(-1).to(X.device)
    col = indices.reshape(-1).to(X.device)
    values = torch.ones_like(row).float().to(X.device)
    attr_simi_mtx = torch.sparse_coo_tensor(torch.stack((row, col),0), values, attr_simi_mtx.shape)
    attr_simi_mtx = attr_simi_mtx.to_dense()
    return attr_simi_mtx



class attr_agg(nn.Module):
    def __init__(self, attr_simi_mtx, alpha, hop, emb_dim, hidden_dim, linear_prop='sgc', linear_trans='lin', norm=None):
        super(attr_agg, self).__init__()

        self.attr_simi_mtx = attr_simi_mtx

        ### setup the linear propagation model ###
        if linear_prop=='sgc':
            I = torch.eye(attr_simi_mtx.shape[0]).to(attr_simi_mtx.device)
            attr_filter = torch.eye(attr_simi_mtx.shape[0]).to(attr_simi_mtx.device)
            for _ in range(hop):
                attr_filter = alpha * attr_simi_mtx @ attr_filter + I
            self.attr_filter = attr_filter
            if linear_trans=='lin':
                self.fc = nn.Linear(emb_dim, emb_dim)
            elif linear_trans=='mlp':
                self.fc = MLP(in_channels=emb_dim, hidden_channels=hidden_dim, out_channels=emb_dim, num_layers=2, batch_norm=False, dropout=0.0, bias=True)
        elif linear_prop=='gcn':
            self.attr_filter = attr_simi_mtx
            self.lins = ModuleList()
            self.lins.append(Linear(in_channels = emb_dim, out_channels = hidden_dim, bias = True, weight_initializer = 'glorot'))
            for i in range(hop-2):
                self.lins.append(Linear(in_channels = hidden_dim, out_channels = hidden_dim, bias = True, weight_initializer = 'glorot'))
            self.lins.append(Linear(in_channels = hidden_dim, out_channels = emb_dim, bias = True, weight_initializer = 'glorot'))
        else:
            raise NotImplementedError
        self.linear_prop = linear_prop
        self.norm = norm

    def agg(self, x):
        return self.attr_filter @ x

    def forward(self, x):
        # x = F.normalize(x, p=2, dim=1)
        if self.linear_prop == 'sgc':
            x = self.fc(x)
            # x = (x - x.mean(0)) / x.std(0) / torch.sqrt(torch.tensor(x.shape[1]).to(x.device))
            if self.norm == 'l2-norm':
                print('l2-norm')
                x = F.normalize(x, p=2, dim=1)
        elif self.linear_prop == 'gcn':
            for lin in self.lins[:-1]:
                x = lin(self.attr_filter @ x)
                x = F.relu(x)
            x = self.lins[-1](self.attr_filter @ x)
        
        return x




class attr_agg_f(nn.Module):
    def __init__(self, half_S_norm, alpha, hop, emb_dim, hidden_dim, linear_prop='sgc', linear_trans='lin', norm=None):
        super(attr_agg_f, self).__init__()
        ### setup the linear propagation model ###
        self.half_S_norm = half_S_norm
        self.alpha = alpha
        self.hop = hop

        if linear_trans=='lin':
            self.fc = nn.Linear(emb_dim, emb_dim)
        elif linear_trans=='mlp':
            self.fc = MLP(in_channels=emb_dim, hidden_channels=hidden_dim, out_channels=emb_dim, num_layers=2, batch_norm=False, dropout=0.0, bias=True)

        self.linear_prop = linear_prop
        self.norm = norm


    def _top_filter(self, x):
        attr_filter = x
        for _ in range(self.hop):
            temp = self.half_S_norm.t() @ attr_filter
            attr_filter = self.alpha * self.half_S_norm @ temp + x
        return attr_filter
    

    def forward(self, x):
        # x = F.normalize(x, p=2, dim=1)
        x = self._top_filter(x)

        x = self.fc(x)
        # x = (x - x.mean(0)) / x.std(0) / torch.sqrt(torch.tensor(x.shape[1]).to(x.device))
        if self.norm == 'l2-norm':
            print('l2-norm')
            x = F.normalize(x, p=2, dim=1)

        
        return x





class fusion(nn.Module):
    def __init__(self, fusion_method, fusion_beta, emb_dim, fusion_norm=None):
        super(fusion, self).__init__()
        self.fusion_method = fusion_method
        self.fusion_beta = fusion_beta
        self.fusion_norm = fusion_norm
        if fusion_method == 'concat':
            self.fusion_fc = nn.Linear(2*emb_dim, emb_dim)
        else: 
            self.fusion_fc = nn.Linear(emb_dim, emb_dim)

    def forward(self, H_low, H_high, beta=None): # an in-place normalization
        if beta is None:
            beta = self.fusion_beta

        H_low = beta * H_low
        H_high = (1-beta) * H_high

        if self.fusion_method == 'add':
            H =  H_low + H_high
        elif self.fusion_method == 'concat':
            H = self.fusion_fc(torch.cat((H_low, H_high), dim=1))
            H = F.normalize(H, p=2, dim=1) 
        elif self.fusion_method == 'max':
            H = torch.max(H_low, H_high)
        else:
            raise NotImplementedError
        
        return H



class C_agg(nn.Module):
    def __init__(self, alpha, hop, A):
        super(C_agg, self).__init__()

        I = torch.eye(A.shape[0]).to(A.device)
        C_filter = torch.eye(A.shape[0]).to(A.device)
        for _ in range(hop):
            C_filter = alpha * torch.spmm(A, C_filter) + I
        # C_filter -= I
        self.C_filter = C_filter

        
    def forward(self, C):
        return self.C_filter @ C
    


class C_agg_f(nn.Module):
    def __init__(self, alpha, hop, A):
        super(C_agg_f, self).__init__()

        self.alpha = alpha
        self.hop = hop
        self.A = A

        
    def forward(self, C):
        C_filter = C
        for _ in range(self.hop):
            C_filter = self.alpha * torch.spmm(self.A, C_filter) + C
        return C_filter




class attr_agg2(nn.Module):
    def __init__(self, X, alpha, hop, input_dim, hidden_dim):
        super(attr_agg2, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
    def forward(self, x):
        return self.fc(x)

class hetero_agg(nn.Module):
    def __init__(self, alpha, hop, A):
        super(hetero_agg, self).__init__()

        self.alpha = alpha
        self.hop = hop
        self.A = A

    def forward(self, C, H):
        A_hetero = self.A - C@C.t()
        A_hetero = F.relu(A_hetero)
        A_hetero = (A_hetero + A_hetero.t())/2

        L_hetero = torch.eye(A_hetero.shape[0]).to(C.device)
        L_hetero = L_hetero * A_hetero.sum(axis=1) - A_hetero

        I = torch.eye(L_hetero.shape[0]).to(C.device)
        filter_hetero = torch.eye(L_hetero.shape[0]).to(C.device)
        for _ in range(self.hop):
            filter_hetero = self.alpha * filter_hetero @ L_hetero + I
        H = F.normalize(filter_hetero @ H, p=2, dim=1)
        return H


    

class cluster_model(nn.Module):
    def __init__(self, method, node_num, hidden_dim, cluster_num, H):
        super(cluster_model, self).__init__()
        self.method = method
        self.node_num = node_num
        self.hidden_dim = hidden_dim
        self.cluster_num = cluster_num

        if method == 'mlp':
            self.fc = nn.Linear(hidden_dim, cluster_num).to(H.device)
            # self.cluster_assi = F.softmax(self.fc(H), dim=1)
        elif method == 'kmeans':
            init = self.kmeans(H)
            self.cluster_assi = nn.Parameter(init)
        elif method == 'random':
            init = torch.rand(node_num, cluster_num)
            self.cluster_assi = nn.Parameter(init)
        else:
            raise NotImplementedError
        
          
    def kmeans(self, H):
        cluster_ids,_ = k_means(H, self.cluster_num, device='gpu', distance='cosine')
        cluster = torch.zeros(H.shape[0], self.cluster_num).to(H.device)
        cluster[torch.arange(H.shape[0]), cluster_ids] = 1
        return cluster
    

    def forward(self, H):
        if self.method == 'mlp':
            return F.softmax(self.fc(H), dim=1)
        else:
            return F.softmax(self.cluster_assi, dim=1)


class low_pass_model(nn.Module):
    # H_0 = MLP(X) or H_0 = linear(X)
    # H = \sum_{i=1}^{L} \alpha_i A^i H_0 W
    def __init__(self, args, low_pass_filter, input_dim, hidden_dim):
        super(low_pass_model, self).__init__()
        self.args = args
        self.low_pass_filter = low_pass_filter
        if args.first_transformation=='mlp':
            self.lin1 = MLP([input_dim, hidden_dim])
        elif args.first_transformation=='linear':
            self.lin1 = nn.Linear(input_dim, hidden_dim)
        else:
            raise NotImplementedError
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X):
        H_0 = self.lin1(X)
        H = self.low_pass_filter @ H_0
        H = self.lin2(H)
        return H



class pool_based_model(nn.Module):
    def __init__(self, encoder, input_dim, hidden_dim, cluster_num, dropout, low_pass_filter, args):
        super(pool_based_model, self).__init__()
        self.encoder = encoder
        if encoder == 'GCN':
            self.encoding = GCNConv(input_dim, hidden_dim)
        elif encoder == 'low_pass':
            self.encoding = low_pass_model(args, low_pass_filter, input_dim, hidden_dim)
        else:
            raise NotImplementedError    
        self.dmon_pool = DMoNPooling(hidden_dim, cluster_num, dropout=dropout)
    
    def forward(self, X, A, edge_index):
        if self.encoder == 'GCN':
            H = self.encoding(X, edge_index)
        elif self.encoder == 'low_pass':
            H = self.encoding(X)
        s, out, out_adj, spectral_loss, ortho_loss, cluster_loss = self.dmon_pool(H, A)
        # remove the dimension for graph level
        return s[0], out[0], out_adj[0], spectral_loss, ortho_loss, cluster_loss


def SSG_CCA_loss_fn(H1, H2):
    loss1 = F.mse_loss(H1, H2)
    # loss2 = F.mse_loss(H1.t() @ H1, torch.eye(H1.shape[1]).to(H1.device))
    # loss3 = F.mse_loss(H2.t() @ H2, torch.eye(H2.shape[1]).to(H2.device))
    # loss = loss1 + 0.1*(loss2 + loss3)
    return loss1

def ortho_loss_fn(H):
    return F.mse_loss(H.t() @ H, torch.eye(H.shape[1]).to(H.device))

def node_t_neighbor_a_loss_fn(H_t, H_a, A):
    H_a = A @ H_a
    loss = F.mse_loss(H_t, H_a)
    return loss

def node_t_neighbor_a_loss_fn2(H_t, H_a, A_no_loop_sym):
    # A_norm = normalize_adj_torch(A_ori, self_loop=False, symmetry=False)
    H_a = torch.spmm(A_no_loop_sym, H_a)
    loss = F.mse_loss(H_t, H_a)
    # loss = (H_t - H_a).norm(p=2, dim=1).mean()

    return loss

def node_t_cluster_a_loss_fn(H_t, H_a, C, simi=None, centers=None):
    if simi is None:
        simi = torch.ones(H_t.shape[0]).to(H_t.device)
    if centers is None:
        C = F.normalize(C, p=2, dim=1)
        centers = C.t() @ H_a # K x d
    predict_labels = torch.argmax(C, dim=1)
    centers = centers[predict_labels]
    # loss = torch.pow((simi * (H_t - centers).norm(p=2, dim=1)), 2).mean()
    loss = (simi * (H_t - centers).norm(p=2, dim=1)).mean()
    return loss


def node_t_cluster_a_loss_fn2(H_t, H_a, C, simi=None, centers=None, clu_size=True):
    if centers is None:
        # print(C.sum(0), C.sum(0).mean())
        if clu_size == False:
            print('clu_size is False')
            C = F.normalize(C, p=1, dim=0)
        # C = C/C.sum(0).mean()
        centers = C.t() @ H_a # K x d
    predict_labels = torch.argmax(C, dim=1)
    centers = centers[predict_labels]
    if simi is None:
        loss = F.mse_loss(H_t, centers)
        # loss = (H_t - centers).norm(p=2, dim=1).mean()
        # loss = (H_t - centers).pow(2).sum(dim=1).mean()
    else:
        loss = (simi * (H_t - centers).pow(2)).mean()
    return loss


def kmeans_loss_fn(H, C, args):
    if args.kmeans_loss == 'tr':
        return node_t_cluster_a_loss_fn(H, H, C)
    elif args.kmeans_loss == 'cen':
        return kmeans_centroid_contrastive_loss_fn(H, C, args)
    elif args.kmeans_loss == 'nod':
        return kmeans_node_contrastive_loss_fn(H, C)
    else:
        raise NotImplementedError


def kmeans_trace_loss_fn(H, C, centers=None):
    if centers is None:
        centers = C.T @ H
    loss = 0
    for i in range(C.shape[1]):
        loss += torch.pow(H[torch.argmax(C, dim=1) == i] - centers[i],2).sum(axis=1).mean()
        # loss += torch.pow(((H - centers[i]) * C[:, i].unsqueeze(1)),2).sum()
    return loss/C.shape[0]/C.shape[1]


def kmeans_centroid_contrastive_loss_fn(H, C, args):
    # pos: node representation vs. its cluster centroid
    # neg: node representation vs. other cluster centroids


    # C_bin = torch.zeros_like(C)
    # pred_clu = torch.argmax(C, dim=1)
    # C_bin[torch.arange(C.shape[0]), pred_clu] = 1
    # C_bin = F.normalize(C_bin, p=1, dim=0)
    # cluster_centroid_embedding = C_bin.T @ H
    C = F.normalize(C, p=1, dim=0)
    cluster_centroid_embedding = C.T @ H

    H = F.normalize(H, p=2, dim=1)
    cluster_centroid_embedding = F.normalize(cluster_centroid_embedding, p=2, dim=1)
    sim = torch.einsum("nd, kd -> nk", H, cluster_centroid_embedding)
    sim = sim * C
    sim /= args.temperature
    labels = torch.argmax(C, dim=1)
    return F.cross_entropy(sim, labels) 

# TODO: implement this function
def density_estimation(H, C, args):
    # concentration estimation (phi)        
    Dcluster = []
    for i in range(C.shape[1]):
        Dcluster.append((H[torch.argmax(C, dim=1)==i] - H[torch.argmax(C, dim=1)==i].mean(1)).norm(p=2, dim=1).list())
    
    
    density = np.zeros(C.shape[1])
    for i,dist in enumerate(Dcluster):
        if len(dist)>1:
            d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
            density[i] = d     
            
    #if cluster only has one point, use the max to estimate its concentration        
    dmax = density.max()
    for i,dist in enumerate(Dcluster):
        if len(dist)<=1:
            density[i] = dmax 

    density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
    density = args.temperature*density/density.mean()  #scale the mean to temperature 

def kmeans_node_contrastive_loss_fn(H, C, num_neg_samples):
    # pos: nodes in the same cluster
    # neg: nodes in different clusters

    labels = torch.argmax(C, dim=1)
    num_nodes = H.shape[0]
    loss = 0

    for i in range(num_nodes):
        target_label = labels[i]
        pos_indices = torch.where(labels == target_label)[0]
        neg_indices = torch.where(labels != target_label)[0]

        # 随机选择一个正样本
        pos_index = pos_indices[torch.randperm(pos_indices.size(0))[0]]
        pos_sample = H[pos_index]

        # 随机选择 num_neg_samples 个负样本
        neg_indices = neg_indices[torch.randperm(neg_indices.size(0))[:num_neg_samples]]
        neg_samples = H[neg_indices]

        # 计算 InfoNCE loss
        pos_sim = torch.dot(H[i], pos_sample)
        neg_sim = torch.matmul(H[i], neg_samples.T)
        logits = torch.cat([pos_sim.unsqueeze(0), neg_sim])
        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=H.device)
        loss += F.cross_entropy(logits, targets)

    return loss / num_nodes


def spectral_loss_fn(C, A):
    temp = torch.matmul(C.t(), A)
    return -torch.trace(torch.matmul(temp, C))


def reg_loss_fn(C, args):
    if args.reg_loss == 'orth':
        # || C^T C - I ||_2^F
        return torch.norm(torch.matmul(C.t(), C)-
                      torch.eye(C.shape[1]).to(C.device), p=2)
    elif args.reg_loss == 'col':
        # sqrt(K)/n ||sum C_i ||_F - 1
        return torch.norm(C.sum(dim=0), p=2) * np.sqrt(C.shape[1])/C.shape[0] -1
    elif args.reg_loss == 'sqrt':
        # -trace(sqrt(C^T C))
        return -torch.trace(torch.sqrt(torch.matmul(C.t(), C)+1e-15))
    else:
        raise NotImplementedError
    


def reconstruction_loss_distance(A, B):
    return torch.norm(A-B, p=2)


def reconstruction_loss_cosine(A, B, beta=1):
    A = F.normalize(A, p=2, dim=1)
    B = F.normalize(B, p=2, dim=1)
    return torch.pow(1 - torch.sum(A*B, dim=1), beta).sum()


def DePropagate(C, C0, A, gamma, alphaC, alphaO):
    z =  (1 - gamma * alphaC + gamma * alphaO) * C
    s = gamma * alphaC * torch.spmm(A, C)
    t = gamma * alphaO * C @ (C.t() @ C)
    return z + s - t + gamma*C0



class input_enc(nn.Module):
    """preprocess S or A, used as input to following model and attr_model
    args:
        model (str): svd on S or A, lin(ear) on S or A, or mlp on S or A
        dims (list): [hidden_dim] for svd, [input_dim, hidden_dim] for lin, [input_dim, hidden_dim1, hidden_dim2, ..., hidden_dimn] for mlp
    """
    def __init__(self, enc, input_dim, hidden_dim, emb_dim):
        super(input_enc, self).__init__()
        self.model = enc
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        if self.model == 'lin':
            self.fc = nn.Linear(input_dim, emb_dim)
        elif self.model == 'mlp':
            self.fc = MLP([input_dim, hidden_dim, emb_dim], batch_norm=False, dropout=0.0, bias=True)
    

    def init_U(self, mtx1, mtx2):
        if self.model == 'svd':
            U, _, _ = torch.svd_lowrank(mtx1, q=self.emb_dim, niter=7)
            self.U = U
        elif self.model == 'lin' or self.model == 'mlp':
            self.U = mtx2
        
        

    def forward(self):
        if self.model == 'svd':
            return self.U
        else:
            return self.fc(self.U)



class pre_process_x(nn.Module):
    """preprocess X, used in the reconstruction loss
    args: 
        model (str): svd on smoothed X, lin(ear) on smoothed X, or mlp on smoothed X
        dims (list): [hidden_dim] for svd, [input_dim, hidden_dim] for lin, [input_dim, hidden_dim1, hidden_dim2, ..., hidden_dimn] for mlp
    """
    def __init__(self, model, dims):

        super(pre_process_x, self).__init__()
        self.model = model
        self.dims = dims
        if model == 'lin':
            self.fc = nn.Linear(dims[0], dims[1])
        elif model == 'mlp':
            self.fc = MLP(dims, batch_norm=False, dropout=0.0, bias=True)
    def forward(self, x):
        if self.model == 'svd':
            U, s, _ = torch.svd_lowrank(x, q=self.dims[0], niter=7)
            return U @ torch.diag(s)
        else:
            return self.fc(x)
        
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels,
                             normalize=True)
        self.conv2 = GCNConv(hidden_channels, out_channels,
                             normalize=True)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x
    
    



    
    
    
    
    



# =========================================================
# TAGSL helper functions
# =========================================================

def tagsl_student_t_assignment(H, centroids, eps=1e-12):
    """
    Student-t soft assignment.

    H:
        [N, d] node embeddings.

    centroids:
        [c, d] learnable cluster centroids.

    Returns:
        S with shape [N, c].
    """
    dist_sq = (
        H.pow(2).sum(dim=1, keepdim=True)
        + centroids.pow(2).sum(dim=1).unsqueeze(0)
        - 2.0 * H @ centroids.T
    ).clamp_min(0.0)

    S = 1.0 / (1.0 + dist_sq)

    return S / S.sum(
        dim=1,
        keepdim=True
    ).clamp_min(eps)


def tagsl_target_distribution(S, eps=1e-12):
    """
    Sharpen the soft assignment distribution.

        T_ij =
            (S_ij^2 / sum_i S_ij)
            ---------------------
            sum_j (S_ij^2 / sum_i S_ij)
    """
    weight = S.pow(2) / S.sum(
        dim=0,
        keepdim=True
    ).clamp_min(eps)

    return weight / weight.sum(
        dim=1,
        keepdim=True
    ).clamp_min(eps)


def tagsl_soft_cross_entropy(
    target,
    prediction,
    eps=1e-12
):
    """
    Cross entropy between two probability matrices.

    target:
        Target distribution T_tgt.

    prediction:
        Student-t assignment S.
    """
    prediction = prediction.clamp_min(eps)

    return -(
        target * prediction.log()
    ).sum(dim=1).mean()


def tagsl_sparse_adjacency_mse(H, A):
    """
    Compute MSE(HH^T, A) without constructing HH^T
    when A is a sparse COO tensor.

    ||HH^T - A||_F^2
        =
    ||H^T H||_F^2
        - 2 Tr(H^T A H)
        + ||A||_F^2
    """
    if not A.is_sparse:
        return F.mse_loss(
            H @ H.T,
            A
        )

    A = A.coalesce()
    num_nodes = H.shape[0]

    HtH = H.T @ H

    loss = (
        HtH.pow(2).sum()
        - 2.0 * (
            H * torch.sparse.mm(A, H)
        ).sum()
        + A.values().pow(2).sum()
    )

    return loss.clamp_min(0.0) / (
        num_nodes * num_nodes
    )


def tagsl_make_F(
    cluster_ids,
    num_clusters,
    dtype
):
    """
    Construct the orthonormal K-means indicator matrix F.

    F^T F = I when every cluster is non-empty.
    """
    cluster_ids = cluster_ids.long()

    F_matrix = torch.zeros(
        cluster_ids.shape[0],
        num_clusters,
        device=cluster_ids.device,
        dtype=dtype
    )

    F_matrix.scatter_(
        1,
        cluster_ids.unsqueeze(1),
        1.0
    )

    cluster_sizes = F_matrix.sum(
        dim=0,
        keepdim=True
    )

    if torch.any(cluster_sizes == 0):
        raise RuntimeError(
            "K-means produced an empty cluster."
        )

    return F_matrix / cluster_sizes.sqrt()



# =========================================================
# TAGSL K-means construction of orthonormal F
# =========================================================

@torch.no_grad()
def tagsl_kmeans_to_F(
    H,
    num_clusters,
    seed=42,
    n_init=10,
    max_iter=300,
    tol=1e-4,
    return_diagnostics=False,
):
    """
    Construct the TAGSL orthonormal indicator matrix F from
    the current structural embedding H using Euclidean K-means.

    Parameters
    ----------
    H:
        Current structural embedding with shape [N, d].

    num_clusters:
        Target number of clusters c.

    seed:
        Fixed random seed passed to sklearn KMeans.

    n_init:
        Number of independent K-means initializations.

    max_iter:
        Maximum number of iterations for one K-means run.

    tol:
        Relative convergence tolerance used by K-means.

    return_diagnostics:
        When False:
            return cluster_ids, F_matrix.

        When True:
            return cluster_ids, F_matrix, diagnostics.

    Returns
    -------
    cluster_ids:
        Long tensor with shape [N], located on H.device.

    F_matrix:
        Orthonormal indicator matrix with shape [N, c],
        located on H.device and using H.dtype.

    Notes
    -----
    - Only the current H is used. Ground-truth labels are never
      accepted or accessed.
    - H is detached and copied to CPU for sklearn K-means.
    - No feature normalization is applied before K-means.
    - Cluster-label permutations do not affect the structural
      objective because F F^T and ||QF||_F are invariant to
      permutations of the columns of F.
    - Every requested cluster must be non-empty. Degenerate
      K-means results are rejected rather than silently repaired.
    """
    import math
    import numbers
    import warnings

    try:
        from sklearn.cluster import KMeans
        from sklearn.exceptions import (
            ConvergenceWarning
        )
    except ImportError as exc:
        raise ImportError(
            "tagsl_kmeans_to_F requires scikit-learn."
        ) from exc

    # -----------------------------------------------------
    # Validate H
    # -----------------------------------------------------
    if not torch.is_tensor(H):
        raise TypeError(
            "H must be a torch.Tensor."
        )

    if H.ndim != 2:
        raise ValueError(
            "H must have shape [N, d]."
        )

    num_nodes, embedding_dim = H.shape

    if num_nodes <= 1:
        raise ValueError(
            "H must contain at least two nodes."
        )

    if embedding_dim <= 0:
        raise ValueError(
            "H must contain at least one embedding "
            "dimension."
        )

    if H.dtype not in {
        torch.float32,
        torch.float64,
    }:
        raise TypeError(
            "H must use torch.float32 or torch.float64."
        )

    if not torch.isfinite(
        H
    ).all():
        raise ValueError(
            "H contains NaN or Inf."
        )

    # -----------------------------------------------------
    # Validate integer arguments
    # -----------------------------------------------------
    if (
        isinstance(num_clusters, bool)
        or not isinstance(
            num_clusters,
            numbers.Integral,
        )
    ):
        raise TypeError(
            "num_clusters must be an integer."
        )

    num_clusters = int(
        num_clusters
    )

    if (
        num_clusters <= 0
        or num_clusters >= num_nodes
    ):
        raise ValueError(
            "num_clusters must satisfy "
            "1 <= num_clusters < num_nodes."
        )

    if (
        isinstance(seed, bool)
        or not isinstance(
            seed,
            numbers.Integral,
        )
    ):
        raise TypeError(
            "seed must be an integer."
        )

    seed = int(seed)

    if (
        seed < 0
        or seed > 2**32 - 1
    ):
        raise ValueError(
            "seed must lie in [0, 2^32 - 1]."
        )

    if (
        isinstance(n_init, bool)
        or not isinstance(
            n_init,
            numbers.Integral,
        )
    ):
        raise TypeError(
            "n_init must be an integer."
        )

    n_init = int(n_init)

    if n_init <= 0:
        raise ValueError(
            "n_init must be positive."
        )

    if (
        isinstance(max_iter, bool)
        or not isinstance(
            max_iter,
            numbers.Integral,
        )
    ):
        raise TypeError(
            "max_iter must be an integer."
        )

    max_iter = int(max_iter)

    if max_iter <= 0:
        raise ValueError(
            "max_iter must be positive."
        )

    if (
        isinstance(tol, bool)
        or not isinstance(
            tol,
            numbers.Real,
        )
    ):
        raise TypeError(
            "tol must be a real scalar."
        )

    tol = float(tol)

    if not math.isfinite(tol):
        raise ValueError(
            "tol must be finite."
        )

    if tol < 0:
        raise ValueError(
            "tol must be nonnegative."
        )

    # -----------------------------------------------------
    # Move a detached copy to CPU.
    #
    # clone() guarantees that sklearn cannot alter the
    # original H storage even when H is already on CPU.
    # -----------------------------------------------------
    H_cpu = (
        H.detach()
        .to(device="cpu")
        .contiguous()
        .clone()
    )

    estimator = KMeans(
        n_clusters=num_clusters,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=seed,
        copy_x=True,
    )

    # Record convergence warnings so a degenerate solution can
    # be converted into a clear runtime error.
    with warnings.catch_warnings(
        record=True
    ) as caught_warnings:
        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )

        labels_numpy = (
            estimator.fit_predict(
                H_cpu.numpy()
            )
        )

    warning_messages = tuple(
        str(warning.message)
        for warning in caught_warnings
    )

    cluster_ids_cpu = torch.from_numpy(
        labels_numpy.copy()
    ).long()

    unique_ids, cluster_sizes = (
        torch.unique(
            cluster_ids_cpu,
            sorted=True,
            return_counts=True,
        )
    )

    expected_ids = torch.arange(
        num_clusters,
        dtype=torch.long,
    )

    # sklearn may return fewer distinct clusters when H
    # contains too few distinct points.
    if (
        unique_ids.numel()
        != num_clusters
        or not torch.equal(
            unique_ids,
            expected_ids,
        )
    ):
        warning_text = (
            "; ".join(
                warning_messages
            )
            if warning_messages
            else "No convergence warning was emitted."
        )

        raise RuntimeError(
            "K-means failed to construct all requested "
            "non-empty clusters. "
            f"Requested={num_clusters}, "
            f"obtained={unique_ids.numel()}. "
            f"Details: {warning_text}"
        )

    cluster_ids = cluster_ids_cpu.to(
        device=H.device,
        dtype=torch.long,
    )

    F_matrix = tagsl_make_F(
        cluster_ids=cluster_ids,
        num_clusters=num_clusters,
        dtype=H.dtype,
    )

    # -----------------------------------------------------
    # Internal verification of F^T F = I.
    # -----------------------------------------------------
    gram = (
        F_matrix.T
        @ F_matrix
    )

    identity = torch.eye(
        num_clusters,
        device=H.device,
        dtype=H.dtype,
    )

    orthogonality_atol = (
        1e-5
        if H.dtype == torch.float32
        else 1e-10
    )

    if not torch.allclose(
        gram,
        identity,
        atol=orthogonality_atol,
        rtol=orthogonality_atol,
    ):
        raise RuntimeError(
            "Internal error: the K-means indicator "
            "matrix does not satisfy F.T @ F = I."
        )

    centers = torch.from_numpy(
        estimator.cluster_centers_.copy()
    ).to(
        device=H.device,
        dtype=H.dtype,
    )

    inertia = torch.as_tensor(
        estimator.inertia_,
        device=H.device,
        dtype=H.dtype,
    )

    if not torch.isfinite(
        inertia
    ).item():
        raise FloatingPointError(
            "K-means returned a non-finite inertia."
        )

    if not torch.isfinite(
        centers
    ).all():
        raise FloatingPointError(
            "K-means returned non-finite centers."
        )

    if not return_diagnostics:
        return (
            cluster_ids,
            F_matrix,
        )

    diagnostics = {
        "inertia": (
            inertia.detach()
        ),
        "n_iter": int(
            estimator.n_iter_
        ),
        "cluster_sizes": (
            cluster_sizes
            .to(H.device)
            .detach()
        ),
        "kmeans_centers": (
            centers.detach()
        ),
        "orthonormal_error": (
            (
                gram - identity
            )
            .norm()
            .detach()
        ),
        "warning_messages": (
            warning_messages
        ),
        "seed": seed,
        "n_init": n_init,
        "max_iter": max_iter,
        "tol": tol,
    }

    return (
        cluster_ids,
        F_matrix,
        diagnostics,
    )




# =========================================================
# TAGSL Stiefel / semi-orthogonal QR retraction
# =========================================================

@torch.no_grad()
def tagsl_stiefel_retraction(
    H_tentative,
    mode="column_strict",
    rank_tolerance=None,
    reject_rank_deficient=True,
    return_diagnostics=False,
):
    """
    Retract a tentative embedding onto a column- or row-
    semi-orthogonal manifold while preserving its shape.

    Parameters
    ----------
    H_tentative:
        Tentative embedding with shape [N, d].

    mode:
        "column_strict":
            Strictly follows the manuscript constraint:

                H^T H = I_d.

            This requires N >= d. An error is raised when
            N < d.

        "shape_preserving":
            Uses column orthogonality when N >= d:

                H^T H = I_d,

            and row orthogonality when N < d:

                H H^T = I_N.

            The returned tensor always retains shape [N, d].

    rank_tolerance:
        Numerical rank threshold applied to abs(diag(R)).
        When None, the threshold is selected using machine
        precision and the scale of R.

    reject_rank_deficient:
        When True, reject a tentative matrix that does not
        have the full rank required by the selected manifold.

        When False, QR is still performed and the numerical
        rank condition is reported through diagnostics.

    return_diagnostics:
        When False:
            return H_retracted.

        When True:
            return H_retracted, diagnostics.

    Notes
    -----
    - The function is intentionally executed under no_grad.
      H is an explicitly updated structural state rather than
      an Adam-optimized parameter.
    - QR signs are canonicalized using diag(R), reducing
      arbitrary sign changes across repeated evaluations.
    - "shape_preserving" with N < d is an implementation
      convention required to retain the embedding dimension.
      It does not equal the manuscript's H^T H = I constraint.
    """
    valid_modes = {
        "column_strict",
        "shape_preserving",
    }

    if mode not in valid_modes:
        raise ValueError(
            f"Unknown orthogonality mode: {mode}. "
            f"Expected one of {sorted(valid_modes)}."
        )

    if not torch.is_tensor(
        H_tentative
    ):
        raise TypeError(
            "H_tentative must be a torch.Tensor."
        )

    if H_tentative.ndim != 2:
        raise ValueError(
            "H_tentative must have shape [N, d]."
        )

    num_nodes, embedding_dim = (
        H_tentative.shape
    )

    if num_nodes <= 0:
        raise ValueError(
            "H_tentative must contain at least one node."
        )

    if embedding_dim <= 0:
        raise ValueError(
            "H_tentative must contain at least one "
            "embedding dimension."
        )

    if H_tentative.dtype not in {
        torch.float32,
        torch.float64,
    }:
        raise TypeError(
            "tagsl_stiefel_retraction currently supports "
            "torch.float32 and torch.float64. Keep the "
            "explicit structural state H in full precision "
            "when using mixed-precision network training."
        )

    if not torch.isfinite(
        H_tentative
    ).all():
        raise ValueError(
            "H_tentative contains NaN or Inf."
        )

    if (
        mode == "column_strict"
        and num_nodes < embedding_dim
    ):
        raise ValueError(
            "column_strict requires num_nodes >= "
            "embedding_dim so that H.T @ H = I can hold. "
            f"Received H with shape "
            f"({num_nodes}, {embedding_dim}). "
            "Use shape_preserving only when the "
            "row-semi-orthogonal fallback is explicitly "
            "intended."
        )

    # -----------------------------------------------------
    # Select the feasible semi-orthogonal orientation.
    # -----------------------------------------------------
    if num_nodes >= embedding_dim:
        orientation = "column"
        qr_input = H_tentative
        required_rank = embedding_dim

    else:
        # This branch is reachable only in shape_preserving
        # mode because column_strict rejected N < d above.
        orientation = "row"
        qr_input = H_tentative.T
        required_rank = num_nodes

    # qr_input is always tall or square:
    #
    # column:
    #     [N, d], N >= d
    #
    # row:
    #     [d, N], d > N
    Q_factor, R_factor = torch.linalg.qr(
        qr_input,
        mode="reduced",
    )

    diagonal = torch.diagonal(
        R_factor,
        offset=0,
    )

    abs_diagonal = diagonal.abs()

    max_abs_diagonal = (
        abs_diagonal.max()
    )

    min_abs_diagonal = (
        abs_diagonal.min()
    )

    # -----------------------------------------------------
    # Numerical rank threshold
    # -----------------------------------------------------
    if rank_tolerance is None:
        matrix_scale = (
            max_abs_diagonal.clamp_min(
                1.0
            )
        )

        tolerance = (
            torch.finfo(
                H_tentative.dtype
            ).eps
            * max(qr_input.shape)
            * matrix_scale
        )

    else:
        tolerance = torch.as_tensor(
            rank_tolerance,
            device=H_tentative.device,
            dtype=H_tentative.dtype,
        )

        if tolerance.ndim != 0:
            raise ValueError(
                "rank_tolerance must be scalar."
            )

        if not torch.isfinite(
            tolerance
        ).item():
            raise ValueError(
                "rank_tolerance must be finite."
            )

        if tolerance.item() < 0:
            raise ValueError(
                "rank_tolerance must be nonnegative."
            )

    estimated_rank = (
        abs_diagonal > tolerance
    ).sum()

    numerically_rank_deficient = (
        estimated_rank.item()
        < required_rank
    )

    if (
        reject_rank_deficient
        and numerically_rank_deficient
    ):
        raise RuntimeError(
            "The tentative embedding is numerically "
            "rank deficient for the selected retraction. "
            f"Required rank={required_rank}, "
            f"estimated rank={estimated_rank.item()}, "
            f"min |diag(R)|="
            f"{min_abs_diagonal.item():.6e}, "
            f"tolerance={tolerance.item():.6e}."
        )

    # -----------------------------------------------------
    # Canonicalize QR signs.
    #
    # Q D and D R represent the same factorization for any
    # diagonal D whose entries are +/-1. Enforcing a
    # nonnegative diagonal of R reduces arbitrary sign flips.
    # -----------------------------------------------------
    signs = torch.sign(
        diagonal
    )

    signs = torch.where(
        signs == 0,
        torch.ones_like(
            signs
        ),
        signs,
    )

    Q_factor = (
        Q_factor
        * signs.unsqueeze(0)
    )

    R_factor = (
        signs.unsqueeze(1)
        * R_factor
    )

    if orientation == "column":
        H_retracted = Q_factor

        identity = torch.eye(
            embedding_dim,
            device=H_tentative.device,
            dtype=H_tentative.dtype,
        )

        orthogonality_residual = (
            H_retracted.T
            @ H_retracted
            - identity
        )

    else:
        H_retracted = Q_factor.T

        identity = torch.eye(
            num_nodes,
            device=H_tentative.device,
            dtype=H_tentative.dtype,
        )

        orthogonality_residual = (
            H_retracted
            @ H_retracted.T
            - identity
        )

    if (
        H_retracted.shape
        != H_tentative.shape
    ):
        raise RuntimeError(
            "Internal retraction error: output shape "
            "does not match the tentative embedding."
        )

    if not torch.isfinite(
        H_retracted
    ).all():
        raise FloatingPointError(
            "The retracted embedding contains NaN or Inf."
        )

    if not return_diagnostics:
        return H_retracted

    diagnostics = {
        "mode": mode,
        "orientation": orientation,
        "num_nodes": int(
            num_nodes
        ),
        "embedding_dim": int(
            embedding_dim
        ),
        "required_rank": int(
            required_rank
        ),
        "estimated_rank": (
            estimated_rank.detach()
        ),
        "numerically_rank_deficient": bool(
            numerically_rank_deficient
        ),
        "rank_tolerance": (
            tolerance.detach()
        ),
        "min_abs_r_diagonal": (
            min_abs_diagonal.detach()
        ),
        "max_abs_r_diagonal": (
            max_abs_diagonal.detach()
        ),
        "orthogonality_error": (
            orthogonality_residual
            .norm()
            .detach()
        ),
        "max_orthogonality_error": (
            orthogonality_residual
            .abs()
            .max()
            .detach()
        ),
        "input_fro_norm": (
            H_tentative
            .norm()
            .detach()
        ),
        "output_fro_norm": (
            H_retracted
            .norm()
            .detach()
        ),
    }

    return (
        H_retracted,
        diagnostics,
    )


# =========================================================
# TAGSL adjacency convention
# =========================================================

@torch.no_grad()
def tagsl_require_symmetric_adjacency(
    A_raw,
    atol=1e-8,
):
    """
    Validate the TAGSL adjacency convention without
    densifying or modifying the graph.

    A_raw must be a square, symmetric sparse COO matrix.
    The same validated adjacency must be used for both
    landmark-tuple generation and topology-support queries.
    """
    if A_raw.layout != torch.sparse_coo:
        raise TypeError(
            "A_raw must be a sparse COO tensor."
        )

    if atol < 0:
        raise ValueError(
            "atol must be nonnegative."
        )

    A_raw = A_raw.coalesce()

    if A_raw.shape[0] != A_raw.shape[1]:
        raise ValueError(
            "A_raw must be a square adjacency matrix."
        )

    asymmetry = (
        A_raw - A_raw.transpose(0, 1)
    ).coalesce()

    if asymmetry.values().numel() > 0:
        max_error = (
            asymmetry.values()
            .abs()
            .max()
            .item()
        )

        if max_error > atol:
            raise ValueError(
                "TAGSL requires a symmetric A_raw. "
                "Symmetrize the graph once during data "
                "preprocessing, then pass the same A_raw "
                "to landmark generation and topology support."
            )

    return A_raw





# =========================================================
# Topology-aware affinity kernel
# =========================================================

class TopologyAwareAffinity(nn.Module):
    """
    Compute affinity values for selected quadruplets.

    quadruplets:
        [B, 4], with each row representing (i, j, k, l).

    support_1 and support_2:
        [B, 4], with columns corresponding to:

        0: (i, j)
        1: (k, l)
        2: (i, k)
        3: (j, l)

    The class only evaluates the requested quadruplets.
    It never constructs or stores a complete fourth-order
    affinity tensor.
    """

    def __init__(
        self,
        sigma=1.0,
        eps=1e-8,
    ):
        super().__init__()

        if sigma <= 0:
            raise ValueError(
                "sigma must be positive."
            )

        if eps <= 0:
            raise ValueError(
                "eps must be positive."
            )

        self.sigma = float(sigma)
        self.eps = float(eps)

    def forward(
        self,
        H,
        quadruplets,
        support_1=None,
        support_2=None,
        mode="topology_aware",
        return_aux=False,
    ):
        quadruplets = quadruplets.to(
            device=H.device,
            dtype=torch.long,
        )

        i, j, k, l = quadruplets.unbind(
            dim=1
        )

        d_ij = (
            H[i] - H[j]
        ).pow(2).sum(dim=1)

        d_kl = (
            H[k] - H[l]
        ).pow(2).sum(dim=1)

        d_ik = (
            H[i] - H[k]
        ).pow(2).sum(dim=1)

        d_jl = (
            H[j] - H[l]
        ).pow(2).sum(dim=1)

        # -------------------------------------------------
        # Pure feature-based affinity.
        #
        # Removing topology modulation should give neutral
        # weights in both the numerator and denominator.
        # -------------------------------------------------
        if mode == "feature_only":
            w_n = torch.ones_like(d_ij)
            w_d = torch.ones_like(d_ij)

        else:
            if (
                support_1 is None
                or support_2 is None
            ):
                raise ValueError(
                    "support_1 and support_2 are required "
                    f"for affinity mode '{mode}'."
                )

            support_1 = support_1.to(
                device=H.device,
                dtype=H.dtype,
            )

            support_2 = support_2.to(
                device=H.device,
                dtype=H.dtype,
            )

            if mode == "one_hop":
                support = support_1

            elif mode == "two_hop":
                support = support_2

            elif mode == "topology_aware":
                support = (
                    support_1 + support_2
                )

            else:
                raise ValueError(
                    f"Unknown affinity mode: {mode}"
                )

            (
                s_ij,
                s_kl,
                s_ik,
                s_jl,
            ) = support.unbind(dim=1)

            # Target-pair joint support.
            w_n = s_ij * s_kl

            # Cross-pair contextual support.
            w_d = s_ik + s_jl

        E_int = w_n * (
            d_ij + d_kl
        )

        E_ext = (
            w_d * (d_ik + d_jl)
            + self.eps
        )

        ratio = (
            E_int / E_ext
        ).clamp_min(0.0)

        affinity = 1.0 - torch.exp(
            -self.sigma * ratio
        )

        if not return_aux:
            return affinity

        return {
            "affinity": affinity,
            "w_n": w_n,
            "w_d": w_d,
            "E_int": E_int,
            "E_ext": E_ext,
            "ratio": ratio,
            "d_ij": d_ij,
            "d_kl": d_kl,
            "d_ik": d_ik,
            "d_jl": d_jl,
        }
        

# =========================================================
# Sparse topology support for selected landmark blocks
# =========================================================

class SparseTopologySupport(nn.Module):
    """
    Query separately normalized A and A^2 supports without
    materializing the complete A^2 matrix.

    For the selected landmark nodes C, the module computes:

        A_columns  = A[:, C]
        A2_columns = A @ A_columns

    followed by separate symmetric normalization:

        A_norm[:, C]
            =
        D1^(-1/2) A[:, C] D1_C^(-1/2)

        A2_norm[:, C]
            =
        D2^(-1/2) A^2[:, C] D2_C^(-1/2)

    where:

        D1 = diag(A 1)
        D2 = diag(A^2 1)

    Only N x |C| column blocks are materialized.
    """

    def __init__(
        self,
        A_raw,
        eps=1e-12,
    ):
        super().__init__()

        if eps <= 0:
            raise ValueError(
                "eps must be positive."
            )

        # Use one shared adjacency convention for both
        # landmark generation and topology-support queries.
        A_raw = tagsl_require_symmetric_adjacency(
            A_raw
        )

        if torch.any(
            A_raw.values() < 0
        ):
            raise ValueError(
                "A_raw must contain nonnegative "
                "edge weights."
            )

        num_nodes = A_raw.shape[0]

        ones = torch.ones(
            num_nodes,
            1,
            device=A_raw.device,
            dtype=A_raw.dtype,
        )

        # -------------------------------------------------
        # Degree of the raw one-hop adjacency:
        #
        #     degree_1 = A 1
        # -------------------------------------------------
        degree_1 = torch.sparse.mm(
            A_raw,
            ones,
        ).squeeze(1)

        # -------------------------------------------------
        # Degree of the raw two-hop adjacency:
        #
        #     degree_2
        #       = A^2 1
        #       = A (A 1)
        #
        # The full A^2 matrix is never constructed.
        # -------------------------------------------------
        degree_2 = torch.sparse.mm(
            A_raw,
            degree_1.unsqueeze(1),
        ).squeeze(1)

        inv_sqrt_degree_1 = (
            self._safe_inverse_sqrt(
                degree_1,
                eps,
            )
        )

        inv_sqrt_degree_2 = (
            self._safe_inverse_sqrt(
                degree_2,
                eps,
            )
        )

        # Keep the buffer name A because the current
        # NystromTensorApproximation validates node counts
        # through self.topology_support.A.
        self.register_buffer(
            "A",
            A_raw,
            persistent=False,
        )

        self.register_buffer(
            "inv_sqrt_degree_1",
            inv_sqrt_degree_1,
            persistent=False,
        )

        self.register_buffer(
            "inv_sqrt_degree_2",
            inv_sqrt_degree_2,
            persistent=False,
        )

    @staticmethod
    def _safe_inverse_sqrt(
        degree,
        eps,
    ):
        """
        Return degree^(-1/2), using zero for isolated nodes.
        """
        result = torch.zeros_like(
            degree
        )

        positive = degree > eps

        result[positive] = degree[
            positive
        ].rsqrt()

        return result

    @torch.no_grad()
    def _selected_columns(
        self,
        selected_nodes,
    ):
        """
        Extract raw A[:, selected_nodes] as a small dense
        column block.

        selected_nodes must be sorted, which is guaranteed by
        torch.unique(..., sorted=True) in prepare().
        """
        selected_nodes = selected_nodes.to(
            device=self.A.device,
            dtype=torch.long,
        )

        if selected_nodes.numel() == 0:
            raise ValueError(
                "selected_nodes cannot be empty."
            )

        row, col = self.A.indices()
        values = self.A.values()

        num_selected = (
            selected_nodes.numel()
        )

        positions = torch.searchsorted(
            selected_nodes,
            col,
        )

        safe_positions = positions.clamp(
            max=num_selected - 1
        )

        matched = (
            positions < num_selected
        ) & (
            selected_nodes[
                safe_positions
            ] == col
        )

        columns = torch.zeros(
            self.A.shape[0],
            num_selected,
            device=self.A.device,
            dtype=values.dtype,
        )

        columns.index_put_(
            (
                row[matched],
                positions[matched],
            ),
            values[matched],
            accumulate=True,
        )

        return columns

    @torch.no_grad()
    def prepare(
        self,
        tuple_block,
    ):
        """
        Prepare one-hop and two-hop support columns for one
        landmark block.

        tuple_block:
            [r, 3], with each row representing (j_a, k_a, l_a).

        The returned cache is reused by both streaming
        Nyström passes.
        """
        tuple_block = tuple_block.to(
            device=self.A.device,
            dtype=torch.long,
        )

        if (
            tuple_block.ndim != 2
            or tuple_block.shape[1] != 3
        ):
            raise ValueError(
                "tuple_block must have shape [r, 3]."
            )

        selected_nodes, inverse = torch.unique(
            tuple_block.reshape(-1),
            sorted=True,
            return_inverse=True,
        )

        tuple_positions = inverse.reshape(
            tuple_block.shape[0],
            3,
        )

        # -------------------------------------------------
        # Raw selected columns of A.
        # -------------------------------------------------
        support_1_columns = (
            self._selected_columns(
                selected_nodes
            )
        )

        # -------------------------------------------------
        # Raw selected columns of A^2:
        #
        #     A^2[:, C] = A @ A[:, C]
        #
        # This must be computed before support_1_columns is
        # normalized in place.
        # -------------------------------------------------
        support_2_columns = torch.sparse.mm(
            self.A,
            support_1_columns,
        )

        # -------------------------------------------------
        # Separate symmetric normalization of A.
        # -------------------------------------------------
        support_1_columns.mul_(
            self.inv_sqrt_degree_1.unsqueeze(1)
        )

        support_1_columns.mul_(
            self.inv_sqrt_degree_1.index_select(
                0,
                selected_nodes,
            ).unsqueeze(0)
        )

        # -------------------------------------------------
        # Separate symmetric normalization of A^2.
        # -------------------------------------------------
        support_2_columns.mul_(
            self.inv_sqrt_degree_2.unsqueeze(1)
        )

        support_2_columns.mul_(
            self.inv_sqrt_degree_2.index_select(
                0,
                selected_nodes,
            ).unsqueeze(0)
        )

        return {
            "tuples": tuple_block,
            "positions": tuple_positions,
            "support_1": support_1_columns,
            "support_2": support_2_columns,
        }

    @staticmethod
    def _query_columns(
        columns,
        node_ids,
        tuples,
        positions,
    ):
        """
        Query supports for:

            (i, j), (k, l), (i, k), (j, l)

        The result uses landmark-major ordering and has shape:

            [number_of_tuples * number_of_nodes, 4]
        """
        node_ids = node_ids.to(
            device=columns.device,
            dtype=torch.long,
        )

        (
            j_position,
            k_position,
            l_position,
        ) = positions.unbind(dim=1)

        (
            j_nodes,
            k_nodes,
            _,
        ) = tuples.unbind(dim=1)

        num_query_nodes = (
            node_ids.numel()
        )

        # [b, s]
        node_rows = columns.index_select(
            0,
            node_ids,
        )

        # [r, b]
        support_ij = node_rows.index_select(
            1,
            j_position,
        ).T

        support_ik = node_rows.index_select(
            1,
            k_position,
        ).T

        # [r]
        support_kl = columns[
            k_nodes,
            l_position,
        ]

        support_jl = columns[
            j_nodes,
            l_position,
        ]

        support_kl = support_kl.unsqueeze(
            1
        ).expand(
            -1,
            num_query_nodes,
        )

        support_jl = support_jl.unsqueeze(
            1
        ).expand(
            -1,
            num_query_nodes,
        )

        # [r, b, 4] -> [r * b, 4]
        return torch.stack(
            (
                support_ij,
                support_kl,
                support_ik,
                support_jl,
            ),
            dim=-1,
        ).reshape(-1, 4)

    @torch.no_grad()
    def query(
        self,
        node_ids,
        support_cache,
    ):
        """
        Return normalized one-hop and two-hop supports for
        the requested node chunk.
        """
        tuples = support_cache[
            "tuples"
        ]

        positions = support_cache[
            "positions"
        ]

        support_1 = self._query_columns(
            columns=support_cache[
                "support_1"
            ],
            node_ids=node_ids,
            tuples=tuples,
            positions=positions,
        )

        support_2 = self._query_columns(
            columns=support_cache[
                "support_2"
            ],
            node_ids=node_ids,
            tuples=tuples,
            positions=positions,
        )

        return support_1, support_2
    
    

# =========================================================
# Streaming separable Nyström approximation
# =========================================================

class NystromTensorApproximation(nn.Module):
    """
    Streaming computation of the landmark structural operator.

    Landmark convention:

        P_n[a, i] = T_{i, j_a, k_a, l_a}

    Supported operator scales
    -------------------------
    none:
        P_n^T P_n H

    divide_by_m:
        (1 / m) P_n^T P_n H

    row_normalize:
        P_hat^T P_hat H, where each row of P_n is
        normalized to unit L2 norm.

    The paper explicitly uses P_n^T P_n H but does not specify
    additional landmark-scale normalization. Therefore, "none"
    remains the default behavior.
    """

    VALID_AFFINITY_MODES = {
        "feature_only",
        "one_hop",
        "two_hop",
        "topology_aware",
    }

    VALID_SCALE_MODES = {
        "none",
        "divide_by_m",
        "row_normalize",
    }

    def __init__(
        self,
        A_support,
        affinity_module,
        landmark_tuples,
        node_chunk_size=2048,
        landmark_chunk_size=16,
        affinity_mode="topology_aware",
        scale_mode="none",
        scale_eps=1e-12,
        max_materialized_elements=20000000,
    ):
        super().__init__()

        landmark_tuples = torch.as_tensor(
            landmark_tuples,
            dtype=torch.long,
        )

        if (
            landmark_tuples.ndim != 2
            or landmark_tuples.shape[1] != 3
        ):
            raise ValueError(
                "landmark_tuples must have shape [m, 3]."
            )

        if landmark_tuples.shape[0] == 0:
            raise ValueError(
                "landmark_tuples cannot be empty."
            )

        if node_chunk_size <= 0:
            raise ValueError(
                "node_chunk_size must be positive."
            )

        if landmark_chunk_size <= 0:
            raise ValueError(
                "landmark_chunk_size must be positive."
            )

        if max_materialized_elements <= 0:
            raise ValueError(
                "max_materialized_elements must be positive."
            )

        if scale_eps <= 0:
            raise ValueError(
                "scale_eps must be positive."
            )

        if affinity_mode not in self.VALID_AFFINITY_MODES:
            raise ValueError(
                "Unknown affinity_mode: "
                f"{affinity_mode}"
            )

        if scale_mode not in self.VALID_SCALE_MODES:
            raise ValueError(
                "Unknown scale_mode: "
                f"{scale_mode}"
            )

        if A_support.layout != torch.sparse_coo:
            raise TypeError(
                "A_support must be a sparse COO tensor."
            )

        if A_support.shape[0] != A_support.shape[1]:
            raise ValueError(
                "A_support must be square."
            )

        num_nodes = A_support.shape[0]

        if landmark_tuples.min().item() < 0:
            raise IndexError(
                "landmark_tuples contain negative indices."
            )

        if landmark_tuples.max().item() >= num_nodes:
            raise IndexError(
                "landmark_tuples contain node indices "
                "outside the graph."
            )

        if torch.any(
            landmark_tuples[:, 0]
            == landmark_tuples[:, 1]
        ):
            raise ValueError(
                "j_a and k_a must be different."
            )

        if torch.any(
            landmark_tuples[:, 0]
            == landmark_tuples[:, 2]
        ):
            raise ValueError(
                "j_a and l_a must be different."
            )

        if torch.any(
            landmark_tuples[:, 1]
            == landmark_tuples[:, 2]
        ):
            raise ValueError(
                "k_a and l_a must be different."
            )

        self.affinity_module = affinity_module

        self.topology_support = SparseTopologySupport(
            A_support
        )

        self.node_chunk_size = int(
            node_chunk_size
        )

        self.landmark_chunk_size = int(
            landmark_chunk_size
        )

        self.affinity_mode = affinity_mode
        self.scale_mode = scale_mode
        self.scale_eps = float(scale_eps)

        self.max_materialized_elements = int(
            max_materialized_elements
        )

        self.register_buffer(
            "landmark_tuples",
            landmark_tuples,
        )

    @property
    def landmark_size(self):
        return self.landmark_tuples.shape[0]

    def _resolve_mode(
        self,
        mode,
    ):
        if mode is None:
            mode = self.affinity_mode

        if mode not in self.VALID_AFFINITY_MODES:
            raise ValueError(
                f"Unknown affinity mode: {mode}"
            )

        return mode

    def _validate_H(
        self,
        H,
    ):
        if H.ndim != 2:
            raise ValueError(
                "H must have shape [N, d]."
            )

        if (
            H.shape[0]
            != self.topology_support.A.shape[0]
        ):
            raise ValueError(
                "H and A_support have different "
                "node counts."
            )

        if (
            H.device
            != self.topology_support.A.device
        ):
            raise RuntimeError(
                "H and A_support must be on the "
                "same device."
            )

    def _make_quadruplets(
        self,
        node_ids,
        tuple_block,
    ):
        """
        Construct quadruplets in landmark-major order.

        For r landmark tuples and b query nodes:

            output shape = [r * b, 4]
        """
        node_ids = node_ids.to(
            device=tuple_block.device,
            dtype=torch.long,
        )

        num_query_nodes = node_ids.numel()
        num_tuples = tuple_block.shape[0]

        i = node_ids.repeat(
            num_tuples
        )

        j = tuple_block[:, 0].repeat_interleave(
            num_query_nodes
        )

        k = tuple_block[:, 1].repeat_interleave(
            num_query_nodes
        )

        l = tuple_block[:, 2].repeat_interleave(
            num_query_nodes
        )

        return torch.stack(
            (i, j, k, l),
            dim=1,
        )

    def _prepare_support_cache(
        self,
        tuple_block,
        mode,
    ):
        """
        feature_only does not construct or query A/A^2 support.
        """
        if mode == "feature_only":
            return None

        return self.topology_support.prepare(
            tuple_block
        )

    def _affinity_chunk(
        self,
        H,
        node_ids,
        tuple_block,
        support_cache,
        mode,
    ):
        """
        Compute one landmark-node affinity block [r, b].
        """
        quadruplets = self._make_quadruplets(
            node_ids=node_ids,
            tuple_block=tuple_block,
        )

        if mode == "feature_only":
            support_1 = None
            support_2 = None

        else:
            if support_cache is None:
                raise RuntimeError(
                    "Topology support cache is missing."
                )

            support_1, support_2 = (
                self.topology_support.query(
                    node_ids=node_ids,
                    support_cache=support_cache,
                )
            )

        affinity = self.affinity_module(
            H=H,
            quadruplets=quadruplets,
            support_1=support_1,
            support_2=support_2,
            mode=mode,
            return_aux=False,
        )

        return affinity.reshape(
            tuple_block.shape[0],
            node_ids.numel(),
        )

    def _scale_materialized_Pn(
        self,
        P_n,
    ):
        """
        Scale a materialized P_n consistently with apply_PtP().
        """
        if self.scale_mode == "none":
            return P_n

        if self.scale_mode == "divide_by_m":
            # Scaling P_n by 1/sqrt(m) gives:
            #
            #   P_scaled^T P_scaled H
            #       =
            #   (1/m) P_n^T P_n H.
            return P_n / (
                self.landmark_size ** 0.5
            )

        # row_normalize
        row_norm = P_n.norm(
            p=2,
            dim=1,
            keepdim=True,
        ).clamp_min(
            self.scale_eps
        )

        return P_n / row_norm

    @torch.no_grad()
    def apply_PtP(
        self,
        H,
        mode=None,
    ):
        """
        Compute the scaled landmark structural operator without
        materializing the complete P_n.
        """
        self._validate_H(H)

        mode = self._resolve_mode(
            mode
        )

        num_nodes = H.shape[0]
        output = torch.zeros_like(H)

        landmark_tuples = (
            self.landmark_tuples.to(
                H.device
            )
        )

        for landmark_start in range(
            0,
            self.landmark_size,
            self.landmark_chunk_size,
        ):
            landmark_end = min(
                landmark_start
                + self.landmark_chunk_size,
                self.landmark_size,
            )

            tuple_block = landmark_tuples[
                landmark_start:landmark_end
            ]

            support_cache = (
                self._prepare_support_cache(
                    tuple_block=tuple_block,
                    mode=mode,
                )
            )

            Z_block = torch.zeros(
                tuple_block.shape[0],
                H.shape[1],
                device=H.device,
                dtype=H.dtype,
            )

            if self.scale_mode == "row_normalize":
                row_squared_norm = torch.zeros(
                    tuple_block.shape[0],
                    device=H.device,
                    dtype=H.dtype,
                )
            else:
                row_squared_norm = None

            # ---------------------------------------------
            # Pass 1:
            #
            #     Z_block = P_block H
            #
            # For row_normalize, also accumulate:
            #
            #     ||P_block[a, :]||_2^2
            # ---------------------------------------------
            for node_start in range(
                0,
                num_nodes,
                self.node_chunk_size,
            ):
                node_end = min(
                    node_start
                    + self.node_chunk_size,
                    num_nodes,
                )

                node_ids = torch.arange(
                    node_start,
                    node_end,
                    device=H.device,
                )

                P_chunk = self._affinity_chunk(
                    H=H,
                    node_ids=node_ids,
                    tuple_block=tuple_block,
                    support_cache=support_cache,
                    mode=mode,
                )

                Z_block.add_(
                    P_chunk
                    @ H[node_start:node_end]
                )

                if row_squared_norm is not None:
                    row_squared_norm.add_(
                        P_chunk.pow(2).sum(dim=1)
                    )

            # For P_hat = D^{-1} P:
            #
            # P_hat^T P_hat H
            #     =
            # P^T D^{-2} P H.
            if row_squared_norm is not None:
                Z_block.div_(
                    row_squared_norm.clamp_min(
                        self.scale_eps
                    ).unsqueeze(1)
                )

            # ---------------------------------------------
            # Pass 2:
            #
            #     output += P_block^T Z_block
            # ---------------------------------------------
            for node_start in range(
                0,
                num_nodes,
                self.node_chunk_size,
            ):
                node_end = min(
                    node_start
                    + self.node_chunk_size,
                    num_nodes,
                )

                node_ids = torch.arange(
                    node_start,
                    node_end,
                    device=H.device,
                )

                P_chunk = self._affinity_chunk(
                    H=H,
                    node_ids=node_ids,
                    tuple_block=tuple_block,
                    support_cache=support_cache,
                    mode=mode,
                )

                output[
                    node_start:node_end
                ].add_(
                    P_chunk.T @ Z_block
                )

            del support_cache

        if self.scale_mode == "divide_by_m":
            output.div_(
                self.landmark_size
            )

        return output

    @torch.no_grad()
    def build_Pn(
        self,
        H,
        node_ids=None,
        mode=None,
        allow_materialize=False,
    ):
        """
        Materialize a scaled P_n only for small-scale analysis.

        For row_normalize, row norms are computed over the
        requested node subset. Full-operator validation should
        therefore use node_ids=None.
        """
        if not allow_materialize:
            raise RuntimeError(
                "P_n materialization is disabled during "
                "training."
            )

        self._validate_H(H)

        mode = self._resolve_mode(
            mode
        )

        if node_ids is None:
            node_ids = torch.arange(
                H.shape[0],
                device=H.device,
            )

        else:
            node_ids = node_ids.to(
                device=H.device,
                dtype=torch.long,
            ).flatten()

            if node_ids.numel() == 0:
                raise ValueError(
                    "node_ids cannot be empty."
                )

            if node_ids.min().item() < 0:
                raise IndexError(
                    "node_ids contain negative indices."
                )

            if node_ids.max().item() >= H.shape[0]:
                raise IndexError(
                    "node_ids contain indices outside H."
                )

        num_elements = (
            self.landmark_size
            * node_ids.numel()
        )

        if (
            num_elements
            > self.max_materialized_elements
        ):
            raise MemoryError(
                f"P_n would contain {num_elements} "
                "elements, exceeding the configured "
                "analysis limit."
            )

        P_n = torch.empty(
            self.landmark_size,
            node_ids.numel(),
            device=H.device,
            dtype=H.dtype,
        )

        landmark_tuples = (
            self.landmark_tuples.to(
                H.device
            )
        )

        for landmark_start in range(
            0,
            self.landmark_size,
            self.landmark_chunk_size,
        ):
            landmark_end = min(
                landmark_start
                + self.landmark_chunk_size,
                self.landmark_size,
            )

            tuple_block = landmark_tuples[
                landmark_start:landmark_end
            ]

            support_cache = (
                self._prepare_support_cache(
                    tuple_block=tuple_block,
                    mode=mode,
                )
            )

            for node_start in range(
                0,
                node_ids.numel(),
                self.node_chunk_size,
            ):
                node_end = min(
                    node_start
                    + self.node_chunk_size,
                    node_ids.numel(),
                )

                query_ids = node_ids[
                    node_start:node_end
                ]

                P_n[
                    landmark_start:landmark_end,
                    node_start:node_end
                ] = self._affinity_chunk(
                    H=H,
                    node_ids=query_ids,
                    tuple_block=tuple_block,
                    support_cache=support_cache,
                    mode=mode,
                )

            del support_cache

        return self._scale_materialized_Pn(
            P_n
        )
    
    
    
    
# =========================================================
# TAGSL landmark-context tuple initialization
# =========================================================

@torch.no_grad()
def tagsl_generate_landmark_tuples(
    A_raw,
    landmark_size,
    seed=42,
    device=None,
):
    """
    Generate landmark-context tuples:

        (j_a, k_a, l_a)

    for the Nyström convention:

        P_n[a, i] = T_{i, j_a, k_a, l_a}.

    Default construction
    --------------------
    1. Local wedge:

           k_a
            |
           j_a
            |
           l_a

       where k_a and l_a are two distinct one-hop
       neighbors of landmark center j_a.

    2. Degree-one fallback:

           j_a -- k_a -- l_a

       where k_a is the only neighbor of j_a and
       l_a is another neighbor of k_a.

    Notes
    -----
    - Self-loops are ignored.
    - A_raw must be a symmetric undirected adjacency matrix.
    - The same validated A_raw is used by tuple generation
      and topology-support queries.
    - The same center j_a may appear in multiple tuples.
    - A complete tuple is never repeated.
    - Reverse copies of the same wedge are not both generated.
    - Tuple indices depend only on graph topology and seed.
    - The current embedding H and node labels are not used.

    The mapping from a topology-aware tensor entry to a row of
    P_n is an implementation convention because the method text
    specifies the separable Nyström basis but does not provide
    an explicit landmark-index sampling rule.
    """
    if landmark_size <= 0:
        raise ValueError(
            "landmark_size must be positive."
        )

    import random
    from collections import defaultdict

    A_raw = tagsl_require_symmetric_adjacency(
        A_raw
    )

    output_device = (
        A_raw.device
        if device is None
        else device
    )

    # -----------------------------------------------------
    # Build an undirected CPU neighbor dictionary.
    #
    # Only nonzero graph support is used for tuple sampling.
    # Actual affinity weights are still obtained later from
    # the normalized A and A^2 support matrices.
    # -----------------------------------------------------
    row, col = A_raw.indices()
    values = A_raw.values()

    valid_edges = (
        (row != col)
        & (values != 0)
    )

    row = row[valid_edges].detach().cpu().tolist()
    col = col[valid_edges].detach().cpu().tolist()

    neighbor_sets = defaultdict(set)

    for u, v in zip(row, col):
        neighbor_sets[u].add(v)

    neighbors = {
        node: tuple(sorted(node_neighbors))
        for node, node_neighbors
        in neighbor_sets.items()
        if len(node_neighbors) > 0
    }

    # -----------------------------------------------------
    # Valid landmark centers:
    #
    # degree >= 2:
    #     a local wedge can be constructed directly.
    #
    # degree == 1:
    #     the unique neighbor must connect to another node,
    #     allowing a length-two chain fallback.
    #
    # isolated nodes are excluded.
    # -----------------------------------------------------
    valid_centers = []

    for j, neighbors_j in neighbors.items():
        if len(neighbors_j) >= 2:
            valid_centers.append(j)
            continue

        if len(neighbors_j) == 1:
            k = neighbors_j[0]

            has_chain_context = any(
                node != j
                for node in neighbors.get(k, ())
            )

            if has_chain_context:
                valid_centers.append(j)

    if len(valid_centers) == 0:
        raise RuntimeError(
            "No valid landmark center can be constructed "
            "from the current graph."
        )

    # Make tuple sampling independent of sparse COO edge order.
    valid_centers.sort()

    rng = random.Random(seed)

    # For degree >= 2, store unordered context pairs so that
    # (j, k, l) and (j, l, k) are not both selected merely
    # as reverse copies of the same local wedge.
    used_wedges = defaultdict(set)

    # For the degree-one chain fallback, k is fixed and only
    # previously unused l nodes need to be tracked.
    used_chain_nodes = defaultdict(set)

    def sample_tuple_for_center(j):
        """
        Return one unused tuple for center j.

        Return None when all valid contexts of j are exhausted.
        """
        neighbors_j = neighbors[j]
        degree_j = len(neighbors_j)

        # -------------------------------------------------
        # Default case: local wedge k - j - l
        # -------------------------------------------------
        if degree_j >= 2:
            pair_count = (
                degree_j * (degree_j - 1) // 2
            )

            # Fast randomized search before deterministic
            # fallback. This avoids enumerating all pairs for
            # high-degree landmark centers.
            random_attempts = min(
                64,
                max(1, pair_count)
            )

            for _ in range(random_attempts):
                k, l = rng.sample(
                    neighbors_j,
                    2
                )

                pair_key = (
                    min(k, l),
                    max(k, l)
                )

                if pair_key in used_wedges[j]:
                    continue

                used_wedges[j].add(pair_key)

                # The affinity is not generally invariant to
                # exchanging k and l. A fixed seed determines
                # one orientation for each sampled wedge.
                if rng.random() < 0.5:
                    k, l = l, k

                return j, k, l

            # Deterministic fallback guarantees that an
            # available pair is found after repeated sampling.
            for first in range(degree_j):
                for second in range(
                    first + 1,
                    degree_j
                ):
                    k = neighbors_j[first]
                    l = neighbors_j[second]

                    pair_key = (k, l)

                    if pair_key in used_wedges[j]:
                        continue

                    used_wedges[j].add(pair_key)

                    if rng.random() < 0.5:
                        k, l = l, k

                    return j, k, l

            return None

        # -------------------------------------------------
        # Degree-one fallback: j - k - l
        # -------------------------------------------------
        k = neighbors_j[0]

        candidate_l = [
            node
            for node in neighbors.get(k, ())
            if (
                node != j
                and node
                not in used_chain_nodes[j]
            )
        ]

        if len(candidate_l) == 0:
            return None

        l = rng.choice(candidate_l)

        used_chain_nodes[j].add(l)

        return j, k, l

    # -----------------------------------------------------
    # Round-robin sampling gives broad center coverage and
    # avoids selecting most tuples from only high-degree hubs.
    #
    # Centers may repeat across rounds, while complete local
    # contexts remain unique.
    # -----------------------------------------------------
    landmark_tuples = []

    while len(landmark_tuples) < landmark_size:
        rng.shuffle(valid_centers)

        generated_in_round = 0

        for j in valid_centers:
            tuple_jkl = sample_tuple_for_center(j)

            if tuple_jkl is None:
                continue

            landmark_tuples.append(tuple_jkl)
            generated_in_round += 1

            if len(landmark_tuples) == landmark_size:
                break

        if generated_in_round == 0:
            break

    if len(landmark_tuples) < landmark_size:
        raise RuntimeError(
            "The graph contains only "
            f"{len(landmark_tuples)} distinct valid "
            "landmark-context tuples under the current "
            f"local wedge/chain rule, but {landmark_size} "
            "tuples were requested."
        )

    return torch.tensor(
        landmark_tuples,
        dtype=torch.long,
        device=output_device,
    )
    
    
# =========================================================
# Dense exact spectral-factor backend
# =========================================================

class DenseExactQBackend(nn.Module):
    """
    Explicit small-graph implementation of the TAGSL
    spectral factor:

        Q in R^{(N-c) x N}
        L = Q^T Q

    The backend manually updates Q according to:

        L(Q)
            =
        alpha ||QH||_F^2
        + lambda ||QF||_F^2
        + rho / 2 ||QF||_F^4.

    Q is stored as a buffer rather than nn.Parameter because
    it is updated by the specialized alternating optimization
    step, not by the outer Adam optimizer.

    This backend requires O(N^2) storage and is intended for
    small-graph verification only.
    """

    VALID_POST_UPDATE_ORDERS = {
        "center_then_dual",
        "dual_then_center",
    }

    def __init__(
        self,
        num_nodes,
        num_clusters,
        alpha=1.0,
        rho=1.0,
        learning_rate=1e-3,
        seed=42,
        dtype=torch.float32,
        device=None,
        post_update_order="center_then_dual",
        max_dense_elements=50000000,
    ):
        super().__init__()

        num_nodes = int(num_nodes)
        num_clusters = int(num_clusters)
        max_dense_elements = int(
            max_dense_elements
        )

        if num_nodes <= 1:
            raise ValueError(
                "num_nodes must be greater than one."
            )

        if (
            num_clusters <= 0
            or num_clusters >= num_nodes
        ):
            raise ValueError(
                "num_clusters must satisfy "
                "1 <= num_clusters < num_nodes."
            )

        if alpha < 0:
            raise ValueError(
                "alpha must be nonnegative."
            )

        if rho <= 0:
            raise ValueError(
                "rho must be positive."
            )

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if max_dense_elements <= 0:
            raise ValueError(
                "max_dense_elements must be positive."
            )

        if (
            post_update_order
            not in self.VALID_POST_UPDATE_ORDERS
        ):
            raise ValueError(
                "Unknown post_update_order: "
                f"{post_update_order}"
            )

        dtype_probe = torch.empty(
            (),
            dtype=dtype,
        )

        if not dtype_probe.is_floating_point():
            raise TypeError(
                "dtype must be a floating-point dtype."
            )

        num_rows = (
            num_nodes - num_clusters
        )

        num_elements = (
            num_rows * num_nodes
        )

        if num_elements > max_dense_elements:
            raise MemoryError(
                "Dense exact Q would require "
                f"{num_elements} elements, exceeding "
                "max_dense_elements="
                f"{max_dense_elements}."
            )

        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.num_rows = num_rows

        self.alpha = float(alpha)
        self.rho = float(rho)

        self.learning_rate = float(
            learning_rate
        )

        self.post_update_order = (
            post_update_order
        )

        self.max_dense_elements = (
            max_dense_elements
        )

        # -------------------------------------------------
        # Initialization convention
        # -------------------------------------------------
        #
        # The manuscript does not provide a concrete random
        # initialization formula for Q.
        #
        # We use deterministic Gaussian initialization with
        # variance scaled by 1 / N, followed by exact
        # row-wise mean-centering.
        # -------------------------------------------------
        generator = torch.Generator()
        generator.manual_seed(
            int(seed)
        )

        Q = torch.randn(
            num_rows,
            num_nodes,
            generator=generator,
            dtype=dtype,
        )

        Q.mul_(
            num_nodes ** -0.5
        )

        Q.sub_(
            Q.mean(
                dim=1,
                keepdim=True,
            )
        )

        if device is not None:
            Q = Q.to(device)

        self.register_buffer(
            "Q",
            Q,
        )

        self.register_buffer(
            "dual_lambda",
            torch.zeros(
                (),
                device=Q.device,
                dtype=Q.dtype,
            ),
        )

    # -----------------------------------------------------
    # Input validation
    # -----------------------------------------------------

    def _validate_H(
        self,
        H,
    ):
        if H.ndim != 2:
            raise ValueError(
                "H must have shape [N, d]."
            )

        if H.shape[0] != self.num_nodes:
            raise ValueError(
                "H has an invalid node dimension."
            )

        if H.device != self.Q.device:
            raise RuntimeError(
                "H and Q must be on the same device."
            )

        if H.dtype != self.Q.dtype:
            raise RuntimeError(
                "H and Q must have the same dtype."
            )

    def _validate_F(
        self,
        F_matrix,
        check_orthonormal=True,
        atol=1e-5,
        rtol=1e-4,
    ):
        expected_shape = (
            self.num_nodes,
            self.num_clusters,
        )

        if (
            F_matrix.ndim != 2
            or F_matrix.shape
            != expected_shape
        ):
            raise ValueError(
                "F_matrix must have shape "
                f"{expected_shape}."
            )

        if F_matrix.device != self.Q.device:
            raise RuntimeError(
                "F_matrix and Q must be on "
                "the same device."
            )

        if F_matrix.dtype != self.Q.dtype:
            raise RuntimeError(
                "F_matrix and Q must have "
                "the same dtype."
            )

        if check_orthonormal:
            gram = (
                F_matrix.T
                @ F_matrix
            )

            identity = torch.eye(
                self.num_clusters,
                device=F_matrix.device,
                dtype=F_matrix.dtype,
            )

            if not torch.allclose(
                gram,
                identity,
                atol=atol,
                rtol=rtol,
            ):
                raise ValueError(
                    "F_matrix must satisfy "
                    "F_matrix.T @ F_matrix = I."
                )

    def _resolve_Q(
        self,
        Q,
    ):
        if Q is None:
            return self.Q

        if Q.shape != self.Q.shape:
            raise ValueError(
                "External Q has an invalid shape."
            )

        if Q.device != self.Q.device:
            raise RuntimeError(
                "External Q and backend Q must be "
                "on the same device."
            )

        if Q.dtype != self.Q.dtype:
            raise RuntimeError(
                "External Q and backend Q must "
                "have the same dtype."
            )

        return Q

    def _resolve_dual_lambda(
        self,
        dual_lambda,
    ):
        if dual_lambda is None:
            return self.dual_lambda

        dual_lambda = torch.as_tensor(
            dual_lambda,
            device=self.Q.device,
            dtype=self.Q.dtype,
        )

        if dual_lambda.ndim != 0:
            raise ValueError(
                "dual_lambda must be scalar."
            )

        return dual_lambda

    # -----------------------------------------------------
    # Eq. (12): augmented Q objective
    # -----------------------------------------------------

    def objective(
        self,
        H,
        F_matrix,
        Q=None,
        dual_lambda=None,
        return_components=False,
        check_orthonormal=True,
    ):
        """
        Evaluate:

            alpha ||QH||_F^2
            + lambda ||QF||_F^2
            + rho/2 ||QF||_F^4.
        """
        self._validate_H(H)

        self._validate_F(
            F_matrix,
            check_orthonormal=(
                check_orthonormal
            ),
        )

        Q_value = self._resolve_Q(
            Q
        )

        lambda_value = (
            self._resolve_dual_lambda(
                dual_lambda
            )
        )

        QH = (
            Q_value @ H
        )

        QF = (
            Q_value @ F_matrix
        )

        h_energy = (
            QH.pow(2).sum()
        )

        violation_sq = (
            QF.pow(2).sum()
        )

        spectral_term = (
            self.alpha
            * h_energy
        )

        dual_term = (
            lambda_value
            * violation_sq
        )

        penalty_term = (
            0.5
            * self.rho
            * violation_sq.pow(2)
        )

        total = (
            spectral_term
            + dual_term
            + penalty_term
        )

        if not return_components:
            return total

        return {
            "total": total,
            "spectral": spectral_term,
            "dual": dual_term,
            "penalty": penalty_term,
            "qf_sq": violation_sq,
        }

    # -----------------------------------------------------
    # Eq. (13): explicit Q gradient
    # -----------------------------------------------------

    def manual_gradient(
        self,
        H,
        F_matrix,
        Q=None,
        dual_lambda=None,
        check_orthonormal=True,
    ):
        """
        Compute:

            2 alpha Q H H^T
            + 2 lambda Q F F^T
            + 2 rho ||QF||_F^2 Q F F^T.

        H H^T and F F^T are not materialized.
        """
        self._validate_H(H)

        self._validate_F(
            F_matrix,
            check_orthonormal=(
                check_orthonormal
            ),
        )

        Q_value = self._resolve_Q(
            Q
        )

        lambda_value = (
            self._resolve_dual_lambda(
                dual_lambda
            )
        )

        QH = (
            Q_value @ H
        )

        QF = (
            Q_value @ F_matrix
        )

        violation_sq = (
            QF.pow(2).sum()
        )

        # Q H H^T
        smoothness_gradient = (
            QH @ H.T
        )

        # Q F F^T
        nullspace_gradient = (
            QF @ F_matrix.T
        )

        return 2.0 * (
            self.alpha
            * smoothness_gradient
            + (
                lambda_value
                + self.rho
                * violation_sq
            )
            * nullspace_gradient
        )

    # -----------------------------------------------------
    # Constraint operations
    # -----------------------------------------------------

    @torch.no_grad()
    def center_rows_(
        self,
    ):
        """
        Enforce:

            Q 1 = 0

        by subtracting the mean of every row.
        """
        self.Q.sub_(
            self.Q.mean(
                dim=1,
                keepdim=True,
            )
        )

        return self

    @torch.no_grad()
    def set_Q_(
        self,
        Q,
        center=False,
    ):
        """
        Replace Q for initialization, restoration or testing.
        """
        if Q.shape != self.Q.shape:
            raise ValueError(
                "Q has an invalid shape."
            )

        if not torch.isfinite(
            Q
        ).all():
            raise ValueError(
                "Q contains NaN or Inf."
            )

        self.Q.copy_(
            Q.to(
                device=self.Q.device,
                dtype=self.Q.dtype,
            )
        )

        if center:
            self.center_rows_()

        return self

    @torch.no_grad()
    def reset_dual_(
        self,
        value=0.0,
    ):
        """
        Reset the scalar dual variable lambda.
        """
        value_tensor = torch.as_tensor(
            value,
            dtype=self.Q.dtype,
        )

        if (
            value_tensor.ndim != 0
            or not torch.isfinite(
                value_tensor
            ).item()
        ):
            raise ValueError(
                "Dual value must be a finite scalar."
            )

        self.dual_lambda.fill_(
            float(value_tensor.item())
        )

        return self

    # -----------------------------------------------------
    # Structural operator
    # -----------------------------------------------------

    def apply_QtQ(
        self,
        H,
    ):
        """
        Compute:

            Q^T Q H

        without materializing Q^T Q.
        """
        self._validate_H(H)

        return self.Q.T @ (
            self.Q @ H
        )

    def structural_energy(
        self,
        H,
    ):
        """
        Compute:

            ||QH||_F^2.
        """
        self._validate_H(H)

        return (
            self.Q @ H
        ).pow(2).sum()

    # -----------------------------------------------------
    # One alternating Q step
    # -----------------------------------------------------

    @torch.no_grad()
    def step(
        self,
        H,
        F_matrix,
        learning_rate=None,
        post_update_order=None,
        check_orthonormal=True,
        return_diagnostics=True,
    ):
        """
        Perform one explicit Q update.

        center_then_dual:
            Eq. (13)
            -> row mean-centering
            -> dual update

        dual_then_center:
            Eq. (13)
            -> dual update
            -> row mean-centering
        """
        self._validate_H(H)

        self._validate_F(
            F_matrix,
            check_orthonormal=(
                check_orthonormal
            ),
        )

        if learning_rate is None:
            learning_rate = (
                self.learning_rate
            )
        else:
            learning_rate = float(
                learning_rate
            )

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if post_update_order is None:
            post_update_order = (
                self.post_update_order
            )

        if (
            post_update_order
            not in self.VALID_POST_UPDATE_ORDERS
        ):
            raise ValueError(
                "Unknown post_update_order: "
                f"{post_update_order}"
            )

        gradient = self.manual_gradient(
            H=H,
            F_matrix=F_matrix,
            check_orthonormal=False,
        )

        if not torch.isfinite(
            gradient
        ).all():
            raise FloatingPointError(
                "The Q gradient contains NaN or Inf."
            )

        gradient_norm = (
            gradient.norm()
        )

        # Eq. (13)
        self.Q.add_(
            gradient,
            alpha=-learning_rate,
        )

        if (
            post_update_order
            == "center_then_dual"
        ):
            self.center_rows_()

            dual_violation_sq = (
                self.Q @ F_matrix
            ).pow(2).sum()

            self.dual_lambda.add_(
                self.rho
                * dual_violation_sq
            )

        else:
            dual_violation_sq = (
                self.Q @ F_matrix
            ).pow(2).sum()

            self.dual_lambda.add_(
                self.rho
                * dual_violation_sq
            )

            self.center_rows_()

        if not torch.isfinite(
            self.Q
        ).all():
            raise FloatingPointError(
                "Updated Q contains NaN or Inf."
            )

        if not torch.isfinite(
            self.dual_lambda
        ).all():
            raise FloatingPointError(
                "Updated dual_lambda contains "
                "NaN or Inf."
            )

        if not return_diagnostics:
            return None

        QF = (
            self.Q @ F_matrix
        )

        return {
            "gradient_norm": (
                gradient_norm.detach()
            ),
            "dual_violation_sq": (
                dual_violation_sq.detach()
            ),
            "qf_sq": (
                QF.pow(2)
                .sum()
                .detach()
            ),
            "q1_norm": (
                self.Q.sum(dim=1)
                .norm()
                .detach()
            ),
            "q_fro_norm": (
                self.Q.norm()
                .detach()
            ),
            "dual_lambda": (
                self.dual_lambda
                .detach()
                .clone()
            ),
        }

    # -----------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------

    @torch.no_grad()
    def constraint_diagnostics(
        self,
        F_matrix,
        include_rank=False,
        check_orthonormal=True,
    ):
        """
        Report constraint and scale diagnostics.

        Rank evaluation is optional because it requires an
        SVD-like computation and is intended for small graphs.
        """
        self._validate_F(
            F_matrix,
            check_orthonormal=(
                check_orthonormal
            ),
        )

        QF = (
            self.Q @ F_matrix
        )

        diagnostics = {
            "qf_sq": (
                QF.pow(2).sum()
            ),
            "qf_norm": (
                QF.norm()
            ),
            "q1_norm": (
                self.Q.sum(dim=1)
                .norm()
            ),
            "q_fro_norm": (
                self.Q.norm()
            ),
        }

        if include_rank:
            diagnostics["rank_Q"] = (
                torch.linalg.matrix_rank(
                    self.Q
                )
            )

        return {
            key: value.detach()
            for key, value
            in diagnostics.items()
        }

    def extra_repr(
        self,
    ):
        return (
            f"num_nodes={self.num_nodes}, "
            f"num_clusters={self.num_clusters}, "
            f"Q_shape=({self.num_rows}, "
            f"{self.num_nodes}), "
            f"alpha={self.alpha}, "
            f"rho={self.rho}, "
            f"learning_rate="
            f"{self.learning_rate}, "
            f"post_update_order="
            f"'{self.post_update_order}'"
        )
        
        
# =========================================================
# Implicit scalable projector backend
# =========================================================

class ImplicitProjectorBackend(nn.Module):
    """
    Scalable structural-operator surrogate based on:

        L_F = I - F F^T,

    where:

        F in R^{N x c}
        F^T F = I.

    The operator is applied without constructing an N x N
    matrix:

        L_F H = H - F (F^T H).

    When F is the normalized cluster-indicator matrix:

        rank(L_F) = N - c,
        L_F F = 0,
        L_F 1 = 0.

    Equivalently, if F_perp is an orthonormal basis of the
    complement of span(F), then:

        Q = F_perp^T
        Q^T Q = I - F F^T.

    This backend is a scalable projector realization. It does
    not perform the explicit augmented-Lagrangian Q update,
    dual-variable update, or row-wise Q centering of Eq. (13).
    """

    def __init__(
        self,
        num_nodes,
        num_clusters,
        dtype=torch.float32,
        device=None,
        require_constant_nullspace=True,
        validation_atol=1e-6,
        validation_rtol=1e-5,
        max_dense_elements=50000000,
    ):
        super().__init__()

        num_nodes = int(num_nodes)
        num_clusters = int(num_clusters)
        max_dense_elements = int(
            max_dense_elements
        )

        if num_nodes <= 1:
            raise ValueError(
                "num_nodes must be greater than one."
            )

        if (
            num_clusters <= 0
            or num_clusters >= num_nodes
        ):
            raise ValueError(
                "num_clusters must satisfy "
                "1 <= num_clusters < num_nodes."
            )

        if validation_atol < 0:
            raise ValueError(
                "validation_atol must be nonnegative."
            )

        if validation_rtol < 0:
            raise ValueError(
                "validation_rtol must be nonnegative."
            )

        if max_dense_elements <= 0:
            raise ValueError(
                "max_dense_elements must be positive."
            )

        dtype_probe = torch.empty(
            (),
            dtype=dtype,
        )

        if not dtype_probe.is_floating_point():
            raise TypeError(
                "dtype must be a floating-point dtype."
            )

        self.num_nodes = num_nodes
        self.num_clusters = num_clusters

        self.require_constant_nullspace = bool(
            require_constant_nullspace
        )

        self.validation_atol = float(
            validation_atol
        )

        self.validation_rtol = float(
            validation_rtol
        )

        self.max_dense_elements = (
            max_dense_elements
        )

        # F is stored as state but is not optimized by Adam.
        self.register_buffer(
            "F_matrix",
            torch.zeros(
                num_nodes,
                num_clusters,
                dtype=dtype,
                device=device,
            ),
        )

        self.register_buffer(
            "_F_initialized",
            torch.tensor(
                False,
                dtype=torch.bool,
                device=device,
            ),
        )

    # -----------------------------------------------------
    # State and validation
    # -----------------------------------------------------

    @property
    def is_initialized(
        self,
    ):
        return bool(
            self._F_initialized.item()
        )

    def _require_initialized(
        self,
    ):
        if not self.is_initialized:
            raise RuntimeError(
                "F_matrix has not been initialized. "
                "Call set_F_() or step() first."
            )

    def _validate_H(
        self,
        H,
    ):
        if H.ndim != 2:
            raise ValueError(
                "H must have shape [N, d]."
            )

        if H.shape[0] != self.num_nodes:
            raise ValueError(
                "H has an invalid node dimension. "
                f"Expected {self.num_nodes}, "
                f"but got {H.shape[0]}."
            )

        if H.device != self.F_matrix.device:
            raise RuntimeError(
                "H and F_matrix must be on the "
                "same device."
            )

        if H.dtype != self.F_matrix.dtype:
            raise RuntimeError(
                "H and F_matrix must have the "
                "same dtype."
            )

        if not torch.isfinite(
            H
        ).all():
            raise ValueError(
                "H contains NaN or Inf."
            )

    def _prepare_F(
        self,
        F_matrix,
    ):
        F_value = torch.as_tensor(
            F_matrix,
            device=self.F_matrix.device,
            dtype=self.F_matrix.dtype,
        )

        expected_shape = (
            self.num_nodes,
            self.num_clusters,
        )

        if (
            F_value.ndim != 2
            or tuple(F_value.shape)
            != expected_shape
        ):
            raise ValueError(
                "F_matrix must have shape "
                f"{expected_shape}."
            )

        if not torch.isfinite(
            F_value
        ).all():
            raise ValueError(
                "F_matrix contains NaN or Inf."
            )

        gram = (
            F_value.T
            @ F_value
        )

        identity = torch.eye(
            self.num_clusters,
            device=F_value.device,
            dtype=F_value.dtype,
        )

        if not torch.allclose(
            gram,
            identity,
            atol=self.validation_atol,
            rtol=self.validation_rtol,
        ):
            gram_error = (
                gram - identity
            ).norm().item()

            raise ValueError(
                "F_matrix must satisfy "
                "F_matrix.T @ F_matrix = I. "
                f"Frobenius error={gram_error:.6e}."
            )

        if self.require_constant_nullspace:
            ones = torch.ones(
                self.num_nodes,
                1,
                device=F_value.device,
                dtype=F_value.dtype,
            )

            reconstructed_ones = (
                F_value
                @ (
                    F_value.T
                    @ ones
                )
            )

            if not torch.allclose(
                reconstructed_ones,
                ones,
                atol=self.validation_atol,
                rtol=self.validation_rtol,
            ):
                residual = (
                    ones
                    - reconstructed_ones
                ).norm().item()

                raise ValueError(
                    "The constant vector must lie in "
                    "span(F_matrix), so that "
                    "(I - F F^T) 1 = 0. "
                    f"Residual norm={residual:.6e}."
                )

        return F_value

    # -----------------------------------------------------
    # F state update
    # -----------------------------------------------------

    @torch.no_grad()
    def set_F_(
        self,
        F_matrix,
    ):
        """
        Store the current orthonormal cluster-indicator basis.
        """
        F_value = self._prepare_F(
            F_matrix
        )

        self.F_matrix.copy_(
            F_value
        )

        self._F_initialized.fill_(
            True
        )

        return self

    @torch.no_grad()
    def clear_F_(
        self,
    ):
        """
        Clear the current projector state.
        """
        self.F_matrix.zero_()

        self._F_initialized.fill_(
            False
        )

        return self

    # -----------------------------------------------------
    # Implicit structural operator
    # -----------------------------------------------------

    def apply_QtQ(
        self,
        H,
    ):
        """
        Unified structural-backend interface.

        For this backend, apply_QtQ(H) means:

            (I - F F^T) H.

        No explicit Q or N x N operator is materialized.
        """
        self._require_initialized()
        self._validate_H(H)

        projected_onto_F = (
            self.F_matrix
            @ (
                self.F_matrix.T
                @ H
            )
        )

        return (
            H - projected_onto_F
        )

    def structural_energy(
        self,
        H,
    ):
        """
        Compute:

            Tr(H^T L_F H),

        where:

            L_F = I - F F^T.

        Since L_F is an orthogonal projector:

            Tr(H^T L_F H)
            =
            ||L_F H||_F^2.
        """
        projected = self.apply_QtQ(
            H
        )

        return (
            H * projected
        ).sum()

    # -----------------------------------------------------
    # Scalable backend update
    # -----------------------------------------------------

    @torch.no_grad()
    def step(
        self,
        H,
        F_matrix,
        return_diagnostics=True,
        include_dense_rank=False,
    ):
        """
        Update the structural operator by replacing the current
        orthonormal basis F.

        Unlike DenseExactQBackend, this operation does not
        perform a gradient update of Q. The current K-means
        indicator basis directly determines:

            L_F = I - F F^T.
        """
        self._validate_H(H)

        self.set_F_(
            F_matrix
        )

        if not return_diagnostics:
            return None

        diagnostics = (
            self.constraint_diagnostics(
                include_dense_rank=(
                    include_dense_rank
                )
            )
        )

        diagnostics[
            "structural_energy"
        ] = (
            self.structural_energy(
                H
            ).detach()
        )

        return diagnostics

    # -----------------------------------------------------
    # Small-graph materialization
    # -----------------------------------------------------

    @torch.no_grad()
    def materialize_operator(
        self,
        allow_materialize=False,
    ):
        """
        Materialize:

            L_F = I - F F^T

        only for small-graph tests and diagnostics.
        """
        self._require_initialized()

        if not allow_materialize:
            raise RuntimeError(
                "Dense operator materialization is disabled. "
                "Pass allow_materialize=True only for "
                "small-graph analysis."
            )

        num_elements = (
            self.num_nodes
            * self.num_nodes
        )

        if (
            num_elements
            > self.max_dense_elements
        ):
            raise MemoryError(
                "The dense structural operator would "
                f"contain {num_elements} elements, "
                "exceeding max_dense_elements="
                f"{self.max_dense_elements}."
            )

        identity = torch.eye(
            self.num_nodes,
            device=self.F_matrix.device,
            dtype=self.F_matrix.dtype,
        )

        return (
            identity
            - self.F_matrix
            @ self.F_matrix.T
        )

    # -----------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------

    @torch.no_grad()
    def constraint_diagnostics(
        self,
        include_dense_rank=False,
    ):
        """
        Report projector properties.

        Dense rank, symmetry and eigenvalue diagnostics are
        optional because they require materializing an N x N
        matrix and are intended only for small graphs.
        """
        self._require_initialized()

        identity_c = torch.eye(
            self.num_clusters,
            device=self.F_matrix.device,
            dtype=self.F_matrix.dtype,
        )

        gram_error = (
            self.F_matrix.T
            @ self.F_matrix
            - identity_c
        ).norm()

        projected_F = self.apply_QtQ(
            self.F_matrix
        )

        ones = torch.ones(
            self.num_nodes,
            1,
            device=self.F_matrix.device,
            dtype=self.F_matrix.dtype,
        )

        projected_ones = self.apply_QtQ(
            ones
        )

        diagnostics = {
            "orthonormal_error": (
                gram_error
            ),
            "f_nullspace_norm": (
                projected_F.norm()
            ),
            "constant_nullspace_norm": (
                projected_ones.norm()
            ),
            "theoretical_rank": torch.tensor(
                self.num_nodes
                - self.num_clusters,
                device=self.F_matrix.device,
                dtype=torch.long,
            ),
        }

        if include_dense_rank:
            operator = (
                self.materialize_operator(
                    allow_materialize=True
                )
            )

            diagnostics.update(
                {
                    "rank_L": (
                        torch.linalg.matrix_rank(
                            operator
                        )
                    ),
                    "symmetry_error": (
                        operator
                        - operator.T
                    ).norm(),
                    "idempotence_error": (
                        operator
                        @ operator
                        - operator
                    ).norm(),
                    "min_eigenvalue": (
                        torch.linalg.eigvalsh(
                            operator
                        ).min()
                    ),
                }
            )

        return {
            key: value.detach()
            for key, value
            in diagnostics.items()
        }

    def extra_repr(
        self,
    ):
        return (
            f"num_nodes={self.num_nodes}, "
            f"num_clusters={self.num_clusters}, "
            f"rank={self.num_nodes - self.num_clusters}, "
            f"require_constant_nullspace="
            f"{self.require_constant_nullspace}, "
            f"initialized={self.is_initialized}"
        )
        


# =========================================================
# Unified TAGSL structural-backend adapter
# =========================================================

class TAGSLStructuralBackend(nn.Module):
    """
    Unified adapter for TAGSL structural operators.

    Supported backends
    ------------------
    dense_exact:
        Explicitly stores

            Q in R^{(N-c) x N}

        and applies:

            Q^T Q H.

        Q is updated with the augmented-Lagrangian gradient
        defined by the TAGSL dense exact formulation.

    implicit_projector:
        Stores the current orthonormal indicator basis F and
        applies:

            (I - F F^T) H.

        This is a scalable projector realization and does not
        perform the explicit Q or dual-variable update.

    The adapter exposes one common interface:

        update(H, F_matrix)
        apply_QtQ(H)
        structural_energy(H)
        diagnostics(F_matrix)
    """

    VALID_BACKEND_TYPES = {
        "dense_exact",
        "implicit_projector",
    }

    def __init__(
        self,
        backend_type,
        num_nodes,
        num_clusters,
        alpha=1.0,
        rho=1.0,
        q_learning_rate=1e-3,
        seed=42,
        dtype=torch.float32,
        device=None,
        q_post_update_order="center_then_dual",
        max_dense_elements=50000000,
        require_constant_nullspace=True,
        validation_atol=1e-6,
        validation_rtol=1e-5,
    ):
        super().__init__()

        backend_type = str(
            backend_type
        )

        if (
            backend_type
            not in self.VALID_BACKEND_TYPES
        ):
            raise ValueError(
                "Unknown structural backend type: "
                f"{backend_type}. "
                "Expected one of "
                f"{sorted(self.VALID_BACKEND_TYPES)}."
            )

        self.backend_type = backend_type
        self.num_nodes = int(num_nodes)
        self.num_clusters = int(
            num_clusters
        )

        if backend_type == "dense_exact":
            self.backend = (
                DenseExactQBackend(
                    num_nodes=num_nodes,
                    num_clusters=num_clusters,
                    alpha=alpha,
                    rho=rho,
                    learning_rate=(
                        q_learning_rate
                    ),
                    seed=seed,
                    dtype=dtype,
                    device=device,
                    post_update_order=(
                        q_post_update_order
                    ),
                    max_dense_elements=(
                        max_dense_elements
                    ),
                )
            )

        else:
            self.backend = (
                ImplicitProjectorBackend(
                    num_nodes=num_nodes,
                    num_clusters=num_clusters,
                    dtype=dtype,
                    device=device,
                    require_constant_nullspace=(
                        require_constant_nullspace
                    ),
                    validation_atol=(
                        validation_atol
                    ),
                    validation_rtol=(
                        validation_rtol
                    ),
                    max_dense_elements=(
                        max_dense_elements
                    ),
                )
            )

    # -----------------------------------------------------
    # Backend properties
    # -----------------------------------------------------

    @property
    def is_dense_exact(
        self,
    ):
        return (
            self.backend_type
            == "dense_exact"
        )

    @property
    def is_implicit_projector(
        self,
    ):
        return (
            self.backend_type
            == "implicit_projector"
        )

    @property
    def is_initialized(
        self,
    ):
        """
        DenseExactQBackend initializes Q in its constructor.

        ImplicitProjectorBackend becomes initialized only
        after receiving its first F matrix.
        """
        if self.is_dense_exact:
            return True

        return self.backend.is_initialized

    @property
    def is_scalable(
        self,
    ):
        """
        Whether the backend avoids O(N^2) state storage.
        """
        return self.is_implicit_projector

    # -----------------------------------------------------
    # Unified structural update
    # -----------------------------------------------------

    @torch.no_grad()
    def update(
        self,
        H,
        F_matrix,
        return_diagnostics=True,
        include_rank=False,
    ):
        """
        Update the selected structural backend.

        dense_exact:
            Performs one explicit Q gradient update,
            row-wise centering and dual update.

        implicit_projector:
            Replaces the current F basis and therefore updates
            the implicit operator I - F F^T.
        """
        if self.is_dense_exact:
            raw_diagnostics = (
                self.backend.step(
                    H=H,
                    F_matrix=F_matrix,
                    return_diagnostics=(
                        return_diagnostics
                    ),
                )
            )

        else:
            raw_diagnostics = (
                self.backend.step(
                    H=H,
                    F_matrix=F_matrix,
                    return_diagnostics=(
                        return_diagnostics
                    ),
                    include_dense_rank=(
                        include_rank
                    ),
                )
            )

        if not return_diagnostics:
            return None

        unified = self.diagnostics(
            H=H,
            F_matrix=F_matrix,
            include_rank=include_rank,
        )

        # Keep backend-specific optimization quantities.
        if self.is_dense_exact:
            unified.update(
                {
                    "gradient_norm": (
                        raw_diagnostics[
                            "gradient_norm"
                        ]
                    ),
                    "dual_violation_sq": (
                        raw_diagnostics[
                            "dual_violation_sq"
                        ]
                    ),
                }
            )

        return unified

    # -----------------------------------------------------
    # Unified structural operator
    # -----------------------------------------------------

    def apply_QtQ(
        self,
        H,
    ):
        """
        Apply the selected structural operator.

        dense_exact:
            Q^T Q H

        implicit_projector:
            (I - F F^T) H
        """
        return self.backend.apply_QtQ(
            H
        )

    def forward(
        self,
        H,
    ):
        """
        Allow:

            output = structural_backend(H)
        """
        return self.apply_QtQ(
            H
        )

    def structural_energy(
        self,
        H,
    ):
        """
        Compute the structural quadratic energy.

        dense_exact:
            ||QH||_F^2

        implicit_projector:
            Tr(H^T (I - F F^T) H)
        """
        return (
            self.backend.structural_energy(
                H
            )
        )

    # -----------------------------------------------------
    # Unified diagnostics
    # -----------------------------------------------------

    @torch.no_grad()
    def diagnostics(
        self,
        H,
        F_matrix=None,
        include_rank=False,
    ):
        """
        Return common diagnostics across both backends.

        Common keys
        -----------
        backend_type:
            String identifying the active backend.

        structural_energy:
            Structural quadratic energy for the supplied H.

        nullspace_violation_sq:
            Dense:
                ||QF||_F^2

            Implicit:
                ||(I - FF^T)F||_F^2

        constant_nullspace_norm:
            Dense:
                ||Q1||_2

            Implicit:
                ||(I - FF^T)1||_2

        rank:
            Included only when include_rank=True.
        """
        if self.is_dense_exact:
            if F_matrix is None:
                raise ValueError(
                    "F_matrix is required for dense-exact "
                    "diagnostics."
                )

            raw = (
                self.backend
                .constraint_diagnostics(
                    F_matrix=F_matrix,
                    include_rank=(
                        include_rank
                    ),
                )
            )

            diagnostics = {
                "backend_type": (
                    self.backend_type
                ),
                "structural_energy": (
                    self.backend
                    .structural_energy(
                        H
                    ).detach()
                ),
                "nullspace_violation_sq": (
                    raw["qf_sq"]
                ),
                "constant_nullspace_norm": (
                    raw["q1_norm"]
                ),
                "q_fro_norm": (
                    raw["q_fro_norm"]
                ),
                "dual_lambda": (
                    self.backend
                    .dual_lambda
                    .detach()
                    .clone()
                ),
            }

            if include_rank:
                diagnostics["rank"] = (
                    raw["rank_Q"]
                )

            return diagnostics

        raw = (
            self.backend
            .constraint_diagnostics(
                include_dense_rank=(
                    include_rank
                )
            )
        )

        diagnostics = {
            "backend_type": (
                self.backend_type
            ),
            "structural_energy": (
                self.backend
                .structural_energy(
                    H
                ).detach()
            ),
            "nullspace_violation_sq": (
                raw[
                    "f_nullspace_norm"
                ].pow(2)
            ),
            "constant_nullspace_norm": (
                raw[
                    "constant_nullspace_norm"
                ]
            ),
            "orthonormal_error": (
                raw[
                    "orthonormal_error"
                ]
            ),
            "theoretical_rank": (
                raw[
                    "theoretical_rank"
                ]
            ),
        }

        if include_rank:
            diagnostics["rank"] = (
                raw["rank_L"]
            )

            diagnostics[
                "symmetry_error"
            ] = raw[
                "symmetry_error"
            ]

            diagnostics[
                "idempotence_error"
            ] = raw[
                "idempotence_error"
            ]

            diagnostics[
                "min_eigenvalue"
            ] = raw[
                "min_eigenvalue"
            ]

        return diagnostics

    # -----------------------------------------------------
    # Optional small-graph state access
    # -----------------------------------------------------

    @torch.no_grad()
    def materialize_operator(
        self,
        allow_materialize=False,
    ):
        """
        Materialize the N x N structural operator only for
        small-graph diagnostics.

        dense_exact:
            Q^T Q

        implicit_projector:
            I - F F^T
        """
        if not allow_materialize:
            raise RuntimeError(
                "Dense operator materialization is disabled. "
                "Pass allow_materialize=True only for "
                "small-graph diagnostics."
            )

        num_elements = (
            self.num_nodes
            * self.num_nodes
        )

        max_dense_elements = (
            self.backend
            .max_dense_elements
        )

        if (
            num_elements
            > max_dense_elements
        ):
            raise MemoryError(
                "The dense structural operator would "
                f"contain {num_elements} elements, "
                "exceeding max_dense_elements="
                f"{max_dense_elements}."
            )

        if self.is_dense_exact:
            return (
                self.backend.Q.T
                @ self.backend.Q
            )

        return (
            self.backend
            .materialize_operator(
                allow_materialize=True
            )
        )

    def extra_repr(
        self,
    ):
        return (
            f"backend_type='{self.backend_type}', "
            f"num_nodes={self.num_nodes}, "
            f"num_clusters={self.num_clusters}, "
            f"scalable={self.is_scalable}, "
            f"initialized={self.is_initialized}"
        )
        
        
# =========================================================
# TAGSL spectral embedding refiner
# =========================================================

class TAGSLSpectralRefiner(nn.Module):
    """
    Explicit alternating update of the TAGSL structural
    embedding H.

    Given the current network projection XW, the structural
    backend L, and the streaming Nyström operator, this class
    computes:

        grad_H
            =
        2 (H - XW)
        + 2 alpha L H
        + 2 beta P_n^T P_n H,

    followed by:

        H_half = H - eta_H grad_H,

    and a QR-based retraction.

    The class manages only the explicit structural state H.
    K-means and construction of F remain outside this class.

    Supported structural backends
    -----------------------------
    dense_exact:
        L H = Q^T Q H.

    implicit_projector:
        L H = (I - F F^T) H.

    Orthogonality modes
    -------------------
    column_strict:
        Requires N >= d and enforces H^T H = I_d.

    shape_preserving:
        Enforces H^T H = I_d when N >= d.
        Enforces H H^T = I_N when N < d.
    """

    VALID_ORTHOGONALITY_MODES = {
        "column_strict",
        "shape_preserving",
    }

    def __init__(
        self,
        num_nodes,
        embedding_dim,
        num_clusters,
        structural_backend,
        nystrom,
        alpha=1.0,
        beta=1.0,
        h_learning_rate=1e-3,
        orthogonality_mode="column_strict",
        reject_rank_deficient=True,
        rank_tolerance=None,
    ):
        super().__init__()

        num_nodes = int(num_nodes)
        embedding_dim = int(
            embedding_dim
        )
        num_clusters = int(
            num_clusters
        )

        if num_nodes <= 1:
            raise ValueError(
                "num_nodes must be greater than one."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive."
            )

        if (
            num_clusters <= 0
            or num_clusters >= num_nodes
        ):
            raise ValueError(
                "num_clusters must satisfy "
                "1 <= num_clusters < num_nodes."
            )

        if alpha < 0:
            raise ValueError(
                "alpha must be nonnegative."
            )

        if beta < 0:
            raise ValueError(
                "beta must be nonnegative."
            )

        if h_learning_rate <= 0:
            raise ValueError(
                "h_learning_rate must be positive."
            )

        if (
            orthogonality_mode
            not in self.VALID_ORTHOGONALITY_MODES
        ):
            raise ValueError(
                "Unknown orthogonality_mode: "
                f"{orthogonality_mode}."
            )

        if (
            orthogonality_mode
            == "column_strict"
            and num_nodes < embedding_dim
        ):
            raise ValueError(
                "column_strict requires "
                "num_nodes >= embedding_dim. "
                f"Received N={num_nodes}, "
                f"d={embedding_dim}."
            )

        if rank_tolerance is not None:
            rank_tolerance = float(
                rank_tolerance
            )

            if (
                not torch.isfinite(
                    torch.tensor(
                        rank_tolerance
                    )
                ).item()
                or rank_tolerance < 0
            ):
                raise ValueError(
                    "rank_tolerance must be a finite "
                    "nonnegative scalar."
                )

        required_backend_methods = [
            "update",
            "apply_QtQ",
            "structural_energy",
            "diagnostics",
        ]

        for method_name in required_backend_methods:
            if not hasattr(
                structural_backend,
                method_name,
            ):
                raise TypeError(
                    "structural_backend is missing "
                    f"required method '{method_name}'."
                )

        if not hasattr(
            nystrom,
            "apply_PtP",
        ):
            raise TypeError(
                "nystrom must provide apply_PtP(H, mode)."
            )

        if hasattr(
            structural_backend,
            "num_nodes",
        ):
            if (
                structural_backend.num_nodes
                != num_nodes
            ):
                raise ValueError(
                    "structural_backend and refiner "
                    "have different node counts."
                )

        if hasattr(
            structural_backend,
            "num_clusters",
        ):
            if (
                structural_backend.num_clusters
                != num_clusters
            ):
                raise ValueError(
                    "structural_backend and refiner "
                    "have different cluster counts."
                )

        nystrom_num_nodes = None

        if hasattr(
            nystrom,
            "num_nodes",
        ):
            nystrom_num_nodes = int(
                nystrom.num_nodes
            )

        elif (
            hasattr(
                nystrom,
                "topology_support",
            )
            and hasattr(
                nystrom.topology_support,
                "A",
            )
        ):
            nystrom_num_nodes = int(
                nystrom.topology_support
                .A.shape[0]
            )

        if (
            nystrom_num_nodes is not None
            and nystrom_num_nodes
            != num_nodes
        ):
            raise ValueError(
                "nystrom and refiner have different "
                "node counts."
            )

        reference_tensor = (
            self._find_backend_reference_tensor(
                structural_backend
            )
        )

        if reference_tensor is None:
            raise TypeError(
                "Cannot infer dtype and device from "
                "structural_backend."
            )

        if reference_tensor.dtype not in {
            torch.float32,
            torch.float64,
        }:
            raise TypeError(
                "The structural backend must use "
                "torch.float32 or torch.float64."
            )

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters

        self.alpha = float(alpha)
        self.beta = float(beta)

        self.h_learning_rate = float(
            h_learning_rate
        )

        self.orthogonality_mode = (
            orthogonality_mode
        )

        self.reject_rank_deficient = bool(
            reject_rank_deficient
        )

        self.rank_tolerance = (
            rank_tolerance
        )

        self.structural_backend = (
            structural_backend
        )

        self.nystrom = nystrom

        self.register_buffer(
            "H_state",
            torch.zeros(
                num_nodes,
                embedding_dim,
                device=reference_tensor.device,
                dtype=reference_tensor.dtype,
            ),
        )

        self.register_buffer(
            "_H_initialized",
            torch.tensor(
                False,
                device=reference_tensor.device,
                dtype=torch.bool,
            ),
        )

    # -----------------------------------------------------
    # Backend and state helpers
    # -----------------------------------------------------

    @staticmethod
    def _find_backend_reference_tensor(
        structural_backend,
    ):
        """
        Find a backend buffer that defines the structural
        state dtype and device.
        """
        if hasattr(
            structural_backend,
            "backend",
        ):
            inner_backend = (
                structural_backend.backend
            )

            if hasattr(
                inner_backend,
                "Q",
            ):
                return inner_backend.Q

            if hasattr(
                inner_backend,
                "F_matrix",
            ):
                return inner_backend.F_matrix

        if hasattr(
            structural_backend,
            "Q",
        ):
            return structural_backend.Q

        if hasattr(
            structural_backend,
            "F_matrix",
        ):
            return structural_backend.F_matrix

        for buffer in (
            structural_backend.buffers()
        ):
            if torch.is_tensor(buffer):
                return buffer

        return None

    @property
    def is_initialized(
        self,
    ):
        return bool(
            self._H_initialized.item()
        )

    def _require_initialized(
        self,
    ):
        if not self.is_initialized:
            raise RuntimeError(
                "H_state has not been initialized. "
                "Call initialize_state() first."
            )

    def _prepare_projected_features(
        self,
        projected_features,
    ):
        """
        Detach XW from the network graph and convert it to the
        full-precision structural-state dtype and device.
        """
        if not torch.is_tensor(
            projected_features
        ):
            raise TypeError(
                "projected_features must be a tensor."
            )

        if projected_features.ndim != 2:
            raise ValueError(
                "projected_features must have "
                "shape [N, d]."
            )

        expected_shape = (
            self.num_nodes,
            self.embedding_dim,
        )

        if (
            tuple(projected_features.shape)
            != expected_shape
        ):
            raise ValueError(
                "projected_features must have shape "
                f"{expected_shape}, but received "
                f"{tuple(projected_features.shape)}."
            )

        prepared = (
            projected_features
            .detach()
            .to(
                device=self.H_state.device,
                dtype=self.H_state.dtype,
            )
        )

        if not torch.isfinite(
            prepared
        ).all():
            raise ValueError(
                "projected_features contain NaN or Inf."
            )

        return prepared

    def _prepare_F(
        self,
        F_matrix,
    ):
        if not torch.is_tensor(
            F_matrix
        ):
            F_matrix = torch.as_tensor(
                F_matrix
            )

        expected_shape = (
            self.num_nodes,
            self.num_clusters,
        )

        if (
            F_matrix.ndim != 2
            or tuple(F_matrix.shape)
            != expected_shape
        ):
            raise ValueError(
                "F_matrix must have shape "
                f"{expected_shape}."
            )

        prepared = (
            F_matrix
            .detach()
            .to(
                device=self.H_state.device,
                dtype=self.H_state.dtype,
            )
        )

        if not torch.isfinite(
            prepared
        ).all():
            raise ValueError(
                "F_matrix contains NaN or Inf."
            )

        return prepared

    def _validate_operator_output(
        self,
        value,
        name,
    ):
        if not torch.is_tensor(value):
            raise TypeError(
                f"{name} must return a tensor."
            )

        if value.shape != self.H_state.shape:
            raise ValueError(
                f"{name} returned shape "
                f"{tuple(value.shape)}, expected "
                f"{tuple(self.H_state.shape)}."
            )

        if value.device != self.H_state.device:
            raise RuntimeError(
                f"{name} returned a tensor on an "
                "incompatible device."
            )

        if value.dtype != self.H_state.dtype:
            raise RuntimeError(
                f"{name} returned a tensor with an "
                "incompatible dtype."
            )

        if not torch.isfinite(
            value
        ).all():
            raise FloatingPointError(
                f"{name} returned NaN or Inf."
            )

    # -----------------------------------------------------
    # H initialization and state access
    # -----------------------------------------------------

    @torch.no_grad()
    def initialize_state(
        self,
        projected_features,
        overwrite=False,
        return_diagnostics=False,
    ):
        """
        Initialize H from the current network projection:

            H^(0) = Retract(XW).
        """
        if (
            self.is_initialized
            and not overwrite
        ):
            raise RuntimeError(
                "H_state is already initialized. "
                "Pass overwrite=True to replace it."
            )

        projected = (
            self._prepare_projected_features(
                projected_features
            )
        )

        H_initial, retraction_diagnostics = (
            tagsl_stiefel_retraction(
                H_tentative=projected,
                mode=self.orthogonality_mode,
                rank_tolerance=(
                    self.rank_tolerance
                ),
                reject_rank_deficient=(
                    self.reject_rank_deficient
                ),
                return_diagnostics=True,
            )
        )

        self.H_state.copy_(
            H_initial
        )

        self._H_initialized.fill_(
            True
        )

        if not return_diagnostics:
            return self.H_state

        return (
            self.H_state,
            {
                "retraction": (
                    retraction_diagnostics
                ),
                "initialized": True,
            },
        )

    @torch.no_grad()
    def clear_state_(
        self,
    ):
        """
        Clear only the explicit H state.

        Structural-backend state is intentionally retained.
        """
        self.H_state.zero_()

        self._H_initialized.fill_(
            False
        )

        return self

    def get_H_state(
        self,
        clone=False,
    ):
        self._require_initialized()

        if clone:
            return (
                self.H_state
                .detach()
                .clone()
            )

        return self.H_state

    # -----------------------------------------------------
    # Structural H gradient
    # -----------------------------------------------------

    @torch.no_grad()
    def compute_gradient(
        self,
        projected_features,
        tensor_mode=None,
        return_terms=False,
    ):
        """
        Compute the explicit H gradient without changing H
        or updating the structural backend.
        """
        self._require_initialized()

        projected = (
            self._prepare_projected_features(
                projected_features
            )
        )

        H_current = self.H_state

        reconstruction_term = (
            H_current - projected
        )

        if self.alpha == 0.0:
            structural_term = (
                torch.zeros_like(
                    H_current
                )
            )

        else:
            structural_term = (
                self.structural_backend
                .apply_QtQ(
                    H_current
                )
            )

            self._validate_operator_output(
                structural_term,
                "structural_backend.apply_QtQ",
            )

        if self.beta == 0.0:
            tensor_term = (
                torch.zeros_like(
                    H_current
                )
            )

        else:
            tensor_term = (
                self.nystrom.apply_PtP(
                    H_current,
                    mode=tensor_mode,
                )
            )

            self._validate_operator_output(
                tensor_term,
                "nystrom.apply_PtP",
            )

        gradient = 2.0 * (
            reconstruction_term
            + self.alpha
            * structural_term
            + self.beta
            * tensor_term
        )

        if not torch.isfinite(
            gradient
        ).all():
            raise FloatingPointError(
                "The H gradient contains NaN or Inf."
            )

        if not return_terms:
            return gradient

        return (
            gradient,
            {
                "reconstruction_term": (
                    reconstruction_term
                ),
                "structural_term": (
                    structural_term
                ),
                "tensor_term": (
                    tensor_term
                ),
            },
        )

    # -----------------------------------------------------
    # One alternating H step
    # -----------------------------------------------------

    @torch.no_grad()
    def step(
        self,
        projected_features,
        F_matrix,
        tensor_mode=None,
        update_backend=True,
        include_rank=False,
        return_diagnostics=True,
    ):
        """
        Perform one TAGSL structural iteration:

        1. Read current H.
        2. Update structural backend using current H and F.
        3. Compute reconstruction, structural and tensor terms.
        4. Take the explicit H gradient step.
        5. Apply QR/semi-orthogonal retraction.
        6. Store the new H state.
        """
        self._require_initialized()

        projected = (
            self._prepare_projected_features(
                projected_features
            )
        )

        F_value = self._prepare_F(
            F_matrix
        )

        H_before = (
            self.H_state.clone()
        )

        backend_diagnostics = None

        if update_backend:
            backend_diagnostics = (
                self.structural_backend.update(
                    H=H_before,
                    F_matrix=F_value,
                    return_diagnostics=(
                        return_diagnostics
                    ),
                    include_rank=(
                        include_rank
                    ),
                )
            )

        else:
            if (
                hasattr(
                    self.structural_backend,
                    "is_initialized",
                )
                and not self.structural_backend
                .is_initialized
            ):
                raise RuntimeError(
                    "The structural backend is not "
                    "initialized and update_backend=False."
                )

            if return_diagnostics:
                backend_diagnostics = (
                    self.structural_backend
                    .diagnostics(
                        H=H_before,
                        F_matrix=F_value,
                        include_rank=(
                            include_rank
                        ),
                    )
                )

        gradient, terms = (
            self.compute_gradient(
                projected_features=projected,
                tensor_mode=tensor_mode,
                return_terms=True,
            )
        )

        H_tentative = (
            H_before
            - self.h_learning_rate
            * gradient
        )

        if not torch.isfinite(
            H_tentative
        ).all():
            raise FloatingPointError(
                "The tentative H update contains "
                "NaN or Inf."
            )

        H_new, retraction_diagnostics = (
            tagsl_stiefel_retraction(
                H_tentative=H_tentative,
                mode=self.orthogonality_mode,
                rank_tolerance=(
                    self.rank_tolerance
                ),
                reject_rank_deficient=(
                    self.reject_rank_deficient
                ),
                return_diagnostics=True,
            )
        )

        self.H_state.copy_(
            H_new
        )

        if not return_diagnostics:
            return self.H_state

        diagnostics = {
            "backend": (
                backend_diagnostics
            ),
            "retraction": (
                retraction_diagnostics
            ),
            "tensor_mode": (
                tensor_mode
            ),
            "alpha": self.alpha,
            "beta": self.beta,
            "h_learning_rate": (
                self.h_learning_rate
            ),
            "gradient_norm": (
                gradient.norm().detach()
            ),
            "reconstruction_term_norm": (
                terms[
                    "reconstruction_term"
                ].norm().detach()
            ),
            "structural_term_norm": (
                terms[
                    "structural_term"
                ].norm().detach()
            ),
            "tensor_term_norm": (
                terms[
                    "tensor_term"
                ].norm().detach()
            ),
            "tentative_step_norm": (
                (
                    H_tentative
                    - H_before
                )
                .norm()
                .detach()
            ),
            "retracted_step_norm": (
                (
                    H_new
                    - H_before
                )
                .norm()
                .detach()
            ),
            "H_fro_norm": (
                H_new.norm().detach()
            ),
        }

        return (
            self.H_state,
            diagnostics,
        )

    # -----------------------------------------------------
    # Current-state diagnostics
    # -----------------------------------------------------

    @torch.no_grad()
    def state_diagnostics(
        self,
        F_matrix=None,
        include_rank=False,
    ):
        self._require_initialized()

        if (
            self.num_nodes
            >= self.embedding_dim
        ):
            orientation = "column"

            identity = torch.eye(
                self.embedding_dim,
                device=self.H_state.device,
                dtype=self.H_state.dtype,
            )

            residual = (
                self.H_state.T
                @ self.H_state
                - identity
            )

        else:
            orientation = "row"

            identity = torch.eye(
                self.num_nodes,
                device=self.H_state.device,
                dtype=self.H_state.dtype,
            )

            residual = (
                self.H_state
                @ self.H_state.T
                - identity
            )

        diagnostics = {
            "initialized": True,
            "orientation": orientation,
            "orthogonality_error": (
                residual.norm().detach()
            ),
            "max_orthogonality_error": (
                residual
                .abs()
                .max()
                .detach()
            ),
            "H_fro_norm": (
                self.H_state
                .norm()
                .detach()
            ),
        }

        backend_initialized = True

        if hasattr(
            self.structural_backend,
            "is_initialized",
        ):
            backend_initialized = (
                self.structural_backend
                .is_initialized
            )

        if backend_initialized:
            prepared_F = None

            if F_matrix is not None:
                prepared_F = (
                    self._prepare_F(
                        F_matrix
                    )
                )

            if (
                getattr(
                    self.structural_backend,
                    "is_dense_exact",
                    False,
                )
                and prepared_F is None
            ):
                diagnostics[
                    "backend"
                ] = None

            else:
                diagnostics[
                    "backend"
                ] = (
                    self.structural_backend
                    .diagnostics(
                        H=self.H_state,
                        F_matrix=prepared_F,
                        include_rank=(
                            include_rank
                        ),
                    )
                )

        else:
            diagnostics["backend"] = None

        return diagnostics

    def extra_repr(
        self,
    ):
        return (
            f"num_nodes={self.num_nodes}, "
            f"embedding_dim={self.embedding_dim}, "
            f"num_clusters={self.num_clusters}, "
            f"alpha={self.alpha}, "
            f"beta={self.beta}, "
            f"h_learning_rate="
            f"{self.h_learning_rate}, "
            f"orthogonality_mode="
            f"'{self.orthogonality_mode}', "
            f"initialized={self.is_initialized}"
        )
        


# =========================================================
# TAGSL topology aggregation branch
# =========================================================

class top_agg_f(nn.Module):
    """
    TAGSL topology aggregation branch.

    The module combines:

        X
        -> linear projection XW
        -> explicit structural state H
        -> K-means construction of F
        -> dynamic structural operator
        -> streaming Nyström tensor update
        -> QR / semi-orthogonal retraction.

    Important state distinction
    ---------------------------
    projected_features:
        XW, retaining the autograd graph. It is used to update
        the learnable projection W through the reconstruction
        loss ||H - XW||_F^2.

    H:
        Explicit alternating-optimization state. It is updated
        by TAGSLSpectralRefiner under no_grad and is not an
        Adam-optimized parameter.

    The forward method returns a dictionary rather than a
    single tensor because subsequent TAGSL losses require both
    projected_features and H.
    """

    VALID_BACKEND_TYPES = {
        "dense_exact",
        "implicit_projector",
    }

    VALID_AFFINITY_MODES = {
        "feature_only",
        "one_hop",
        "two_hop",
        "topology_aware",
    }

    VALID_SCALE_MODES = {
        "none",
        "divide_by_m",
        "row_normalize",
    }

    VALID_ORTHOGONALITY_MODES = {
        "column_strict",
        "shape_preserving",
    }

    def __init__(
        self,
        A_raw,
        input_dim,
        embedding_dim,
        num_clusters,
        alpha=1.0,
        beta=1.0,
        landmark_size=256,
        structural_backend_type="implicit_projector",
        affinity_mode="topology_aware",
        scale_mode="none",
        sigma=1.0,
        affinity_eps=1e-8,
        node_chunk_size=2048,
        landmark_chunk_size=32,
        q_rho=1.0,
        q_learning_rate=1e-3,
        h_learning_rate=1e-3,
        q_post_update_order="center_then_dual",
        orthogonality_mode="shape_preserving",
        kmeans_n_init=10,
        kmeans_max_iter=300,
        kmeans_tol=1e-4,
        projection_bias=True,
        seed=42,
        max_dense_elements=50000000,
        max_materialized_elements=50000000,
        require_constant_nullspace=True,
        validation_atol=1e-6,
        validation_rtol=1e-5,
        reject_rank_deficient=True,
        rank_tolerance=None,
    ):
        super().__init__()

        # -------------------------------------------------
        # Basic scalar validation
        # -------------------------------------------------
        input_dim = int(
            input_dim
        )

        embedding_dim = int(
            embedding_dim
        )

        num_clusters = int(
            num_clusters
        )

        landmark_size = int(
            landmark_size
        )

        node_chunk_size = int(
            node_chunk_size
        )

        landmark_chunk_size = int(
            landmark_chunk_size
        )

        seed = int(
            seed
        )

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be positive."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive."
            )

        if landmark_size <= 0:
            raise ValueError(
                "landmark_size must be positive."
            )

        if node_chunk_size <= 0:
            raise ValueError(
                "node_chunk_size must be positive."
            )

        if landmark_chunk_size <= 0:
            raise ValueError(
                "landmark_chunk_size must be positive."
            )

        if alpha < 0:
            raise ValueError(
                "alpha must be nonnegative."
            )

        if beta < 0:
            raise ValueError(
                "beta must be nonnegative."
            )

        if sigma <= 0:
            raise ValueError(
                "sigma must be positive."
            )

        if affinity_eps <= 0:
            raise ValueError(
                "affinity_eps must be positive."
            )

        if q_rho <= 0:
            raise ValueError(
                "q_rho must be positive."
            )

        if q_learning_rate <= 0:
            raise ValueError(
                "q_learning_rate must be positive."
            )

        if h_learning_rate <= 0:
            raise ValueError(
                "h_learning_rate must be positive."
            )

        if kmeans_n_init <= 0:
            raise ValueError(
                "kmeans_n_init must be positive."
            )

        if kmeans_max_iter <= 0:
            raise ValueError(
                "kmeans_max_iter must be positive."
            )

        if kmeans_tol < 0:
            raise ValueError(
                "kmeans_tol must be nonnegative."
            )

        if seed < 0:
            raise ValueError(
                "seed must be nonnegative."
            )

        # -------------------------------------------------
        # Configuration validation
        # -------------------------------------------------
        structural_backend_type = str(
            structural_backend_type
        )

        affinity_mode = str(
            affinity_mode
        )

        scale_mode = str(
            scale_mode
        )

        orthogonality_mode = str(
            orthogonality_mode
        )

        if (
            structural_backend_type
            not in self.VALID_BACKEND_TYPES
        ):
            raise ValueError(
                "Unknown structural_backend_type: "
                f"{structural_backend_type}."
            )

        if (
            affinity_mode
            not in self.VALID_AFFINITY_MODES
        ):
            raise ValueError(
                "Unknown affinity_mode: "
                f"{affinity_mode}."
            )

        if (
            scale_mode
            not in self.VALID_SCALE_MODES
        ):
            raise ValueError(
                "Unknown scale_mode: "
                f"{scale_mode}."
            )

        if (
            orthogonality_mode
            not in self.VALID_ORTHOGONALITY_MODES
        ):
            raise ValueError(
                "Unknown orthogonality_mode: "
                f"{orthogonality_mode}."
            )

        # -------------------------------------------------
        # Validate and freeze the graph convention.
        #
        # A_raw must be the original symmetric adjacency used
        # consistently by:
        #
        #   landmark generation,
        #   one-hop support,
        #   two-hop support.
        #
        # It must not be replaced by only A_norm.
        # -------------------------------------------------
        A_raw = tagsl_require_symmetric_adjacency(
            A_raw
        )

        if A_raw.dtype not in {
            torch.float32,
            torch.float64,
        }:
            raise TypeError(
                "A_raw must use torch.float32 or "
                "torch.float64."
            )

        if torch.any(
            A_raw.values() < 0
        ):
            raise ValueError(
                "A_raw must contain nonnegative "
                "edge weights."
            )

        num_nodes = int(
            A_raw.shape[0]
        )

        if (
            num_clusters <= 0
            or num_clusters >= num_nodes
        ):
            raise ValueError(
                "num_clusters must satisfy "
                "1 <= num_clusters < num_nodes."
            )

        if (
            orthogonality_mode
            == "column_strict"
            and num_nodes < embedding_dim
        ):
            raise ValueError(
                "column_strict requires "
                "num_nodes >= embedding_dim. "
                f"Received N={num_nodes}, "
                f"d={embedding_dim}."
            )

        # -------------------------------------------------
        # Public configuration
        # -------------------------------------------------
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters

        self.alpha = float(
            alpha
        )

        self.beta = float(
            beta
        )

        self.landmark_size = (
            landmark_size
        )

        self.structural_backend_type = (
            structural_backend_type
        )

        self.affinity_mode = (
            affinity_mode
        )

        self.scale_mode = (
            scale_mode
        )

        self.sigma = float(
            sigma
        )

        self.seed = seed

        self.kmeans_n_init = int(
            kmeans_n_init
        )

        self.kmeans_max_iter = int(
            kmeans_max_iter
        )

        self.kmeans_tol = float(
            kmeans_tol
        )

        self.orthogonality_mode = (
            orthogonality_mode
        )

        # -------------------------------------------------
        # Learnable projection:
        #
        #     XW
        #
        # This is the only trainable component created by the
        # topology branch at this stage.
        # -------------------------------------------------
        self.projection = nn.Linear(
            in_features=input_dim,
            out_features=embedding_dim,
            bias=bool(projection_bias),
            device=A_raw.device,
            dtype=A_raw.dtype,
        )

        # -------------------------------------------------
        # Fixed landmark-index initialization.
        #
        # Landmark tuples depend only on the observed graph
        # topology and the random seed, not on H or labels.
        # -------------------------------------------------
        landmark_tuples = (
            tagsl_generate_landmark_tuples(
                A_raw=A_raw,
                landmark_size=landmark_size,
                seed=seed,
                device=A_raw.device,
            )
        )

        # -------------------------------------------------
        # Topology-aware affinity
        # -------------------------------------------------
        self.affinity = (
            TopologyAwareAffinity(
                sigma=sigma,
                eps=affinity_eps,
            )
        )

        # -------------------------------------------------
        # Streaming Nyström operator
        # -------------------------------------------------
        self.nystrom = (
            NystromTensorApproximation(
                A_support=A_raw,
                affinity_module=(
                    self.affinity
                ),
                landmark_tuples=(
                    landmark_tuples
                ),
                node_chunk_size=(
                    node_chunk_size
                ),
                landmark_chunk_size=(
                    landmark_chunk_size
                ),
                affinity_mode=(
                    affinity_mode
                ),
                scale_mode=(
                    scale_mode
                ),
                max_materialized_elements=(
                    max_materialized_elements
                ),
            )
        )

        # -------------------------------------------------
        # Dynamic structural operator.
        #
        # The same alpha is supplied to the Q block and the H
        # block, preventing the two alternating updates from
        # optimizing differently weighted objectives.
        # -------------------------------------------------
        self.structural_backend = (
            TAGSLStructuralBackend(
                backend_type=(
                    structural_backend_type
                ),
                num_nodes=num_nodes,
                num_clusters=num_clusters,
                alpha=alpha,
                rho=q_rho,
                q_learning_rate=(
                    q_learning_rate
                ),
                seed=seed,
                dtype=A_raw.dtype,
                device=A_raw.device,
                q_post_update_order=(
                    q_post_update_order
                ),
                max_dense_elements=(
                    max_dense_elements
                ),
                require_constant_nullspace=(
                    require_constant_nullspace
                ),
                validation_atol=(
                    validation_atol
                ),
                validation_rtol=(
                    validation_rtol
                ),
            )
        )

        # -------------------------------------------------
        # Explicit H block
        # -------------------------------------------------
        self.refiner = (
            TAGSLSpectralRefiner(
                num_nodes=num_nodes,
                embedding_dim=embedding_dim,
                num_clusters=num_clusters,
                structural_backend=(
                    self.structural_backend
                ),
                nystrom=self.nystrom,
                alpha=alpha,
                beta=beta,
                h_learning_rate=(
                    h_learning_rate
                ),
                orthogonality_mode=(
                    orthogonality_mode
                ),
                reject_rank_deficient=(
                    reject_rank_deficient
                ),
                rank_tolerance=(
                    rank_tolerance
                ),
            )
        )

        # -------------------------------------------------
        # Last K-means state.
        #
        # Fixed-size buffers are used so checkpoint loading
        # does not suffer from empty-buffer shape mismatch.
        # -------------------------------------------------
        self.register_buffer(
            "cluster_ids_state",
            torch.full(
                (
                    num_nodes,
                ),
                fill_value=-1,
                device=A_raw.device,
                dtype=torch.long,
            ),
        )

        self.register_buffer(
            "F_state",
            torch.zeros(
                num_nodes,
                num_clusters,
                device=A_raw.device,
                dtype=A_raw.dtype,
            ),
        )

        self.register_buffer(
            "_cluster_state_initialized",
            torch.tensor(
                False,
                device=A_raw.device,
                dtype=torch.bool,
            ),
        )

        self.register_buffer(
            "_structure_step",
            torch.zeros(
                (),
                device=A_raw.device,
                dtype=torch.long,
            ),
        )

    # -----------------------------------------------------
    # State properties
    # -----------------------------------------------------

    @property
    def is_H_initialized(
        self,
    ):
        return self.refiner.is_initialized

    @property
    def is_cluster_state_initialized(
        self,
    ):
        return bool(
            self._cluster_state_initialized
            .item()
        )

    @property
    def structure_step(
        self,
    ):
        return int(
            self._structure_step.item()
        )

    # -----------------------------------------------------
    # Input validation
    # -----------------------------------------------------

    def _validate_input(
        self,
        x,
    ):
        if not torch.is_tensor(x):
            raise TypeError(
                "x must be a torch.Tensor."
            )

        if x.ndim != 2:
            raise ValueError(
                "x must have shape [N, input_dim]."
            )

        expected_shape = (
            self.num_nodes,
            self.input_dim,
        )

        if (
            tuple(x.shape)
            != expected_shape
        ):
            raise ValueError(
                "x must have shape "
                f"{expected_shape}, but received "
                f"{tuple(x.shape)}."
            )

        projection_weight = (
            self.projection.weight
        )

        if x.device != projection_weight.device:
            raise RuntimeError(
                "x and top_agg_f must be on the "
                "same device."
            )

        if x.dtype != projection_weight.dtype:
            raise RuntimeError(
                "x and projection weights must have "
                "the same dtype."
            )

        if not torch.isfinite(
            x.detach()
        ).all():
            raise ValueError(
                "x contains NaN or Inf."
            )

    # -----------------------------------------------------
    # K-means state
    # -----------------------------------------------------

    @torch.no_grad()
    def _update_cluster_state(
        self,
        H_current,
        return_diagnostics,
    ):
        if return_diagnostics:
            (
                cluster_ids,
                F_matrix,
                kmeans_diagnostics,
            ) = tagsl_kmeans_to_F(
                H=H_current,
                num_clusters=(
                    self.num_clusters
                ),
                seed=self.seed,
                n_init=(
                    self.kmeans_n_init
                ),
                max_iter=(
                    self.kmeans_max_iter
                ),
                tol=self.kmeans_tol,
                return_diagnostics=True,
            )

        else:
            (
                cluster_ids,
                F_matrix,
            ) = tagsl_kmeans_to_F(
                H=H_current,
                num_clusters=(
                    self.num_clusters
                ),
                seed=self.seed,
                n_init=(
                    self.kmeans_n_init
                ),
                max_iter=(
                    self.kmeans_max_iter
                ),
                tol=self.kmeans_tol,
                return_diagnostics=False,
            )

            kmeans_diagnostics = None

        self.cluster_ids_state.copy_(
            cluster_ids
        )

        self.F_state.copy_(
            F_matrix
        )

        self._cluster_state_initialized.fill_(
            True
        )

        return (
            cluster_ids,
            F_matrix,
            kmeans_diagnostics,
        )

    # -----------------------------------------------------
    # Main forward
    # -----------------------------------------------------

    def forward(
        self,
        x,
        update_structure=True,
        tensor_mode=None,
        include_rank=False,
        return_diagnostics=True,
    ):
        """
        Run the TAGSL topology branch.

        Parameters
        ----------
        x:
            Input feature matrix with shape [N, input_dim].

        update_structure:
            When True:
                run K-means on the current H, update the
                structural backend, update H, and retract H.

            When False:
                compute only the learnable projection XW and
                reuse the current structural state.

        tensor_mode:
            Optional per-forward affinity-mode override.
            When None, self.affinity_mode is used.

        include_rank:
            Enable expensive small-graph rank diagnostics.

        return_diagnostics:
            Include initialization, K-means and refiner
            diagnostics in the returned dictionary.
        """
        self._validate_input(
            x
        )

        if tensor_mode is None:
            tensor_mode = (
                self.affinity_mode
            )

        if (
            tensor_mode
            not in self.VALID_AFFINITY_MODES
        ):
            raise ValueError(
                "Unknown tensor_mode: "
                f"{tensor_mode}."
            )

        # -------------------------------------------------
        # Learnable XW path.
        #
        # projected_features retains autograd and must be
        # returned unchanged for the W/network update.
        # -------------------------------------------------
        projected_features = (
            self.projection(
                x
            )
        )

        if not torch.isfinite(
            projected_features.detach()
        ).all():
            raise FloatingPointError(
                "projected_features contain NaN or Inf."
            )

        initialization_diagnostics = None

        # -------------------------------------------------
        # H initialization:
        #
        #     H^(0) = Retract(XW).
        # -------------------------------------------------
        if not self.refiner.is_initialized:
            if return_diagnostics:
                (
                    _,
                    initialization_diagnostics,
                ) = self.refiner.initialize_state(
                    projected_features=(
                        projected_features
                    ),
                    return_diagnostics=True,
                )

            else:
                self.refiner.initialize_state(
                    projected_features=(
                        projected_features
                    ),
                    return_diagnostics=False,
                )

        kmeans_diagnostics = None
        refiner_diagnostics = None

        # -------------------------------------------------
        # One alternating structural update.
        #
        # F is constructed from the current H before the H
        # block is updated.
        # -------------------------------------------------
        if update_structure:
            H_before = (
                self.refiner.get_H_state(
                    clone=False
                )
            )

            (
                cluster_ids,
                F_matrix,
                kmeans_diagnostics,
            ) = self._update_cluster_state(
                H_current=H_before,
                return_diagnostics=(
                    return_diagnostics
                ),
            )

            if return_diagnostics:
                (
                    H_state,
                    refiner_diagnostics,
                ) = self.refiner.step(
                    projected_features=(
                        projected_features
                    ),
                    F_matrix=F_matrix,
                    tensor_mode=tensor_mode,
                    update_backend=True,
                    include_rank=(
                        include_rank
                    ),
                    return_diagnostics=True,
                )

            else:
                H_state = self.refiner.step(
                    projected_features=(
                        projected_features
                    ),
                    F_matrix=F_matrix,
                    tensor_mode=tensor_mode,
                    update_backend=True,
                    include_rank=False,
                    return_diagnostics=False,
                )

            self._structure_step.add_(
                1
            )

        else:
            H_state = (
                self.refiner.get_H_state(
                    clone=False
                )
            )

            if self.is_cluster_state_initialized:
                cluster_ids = (
                    self.cluster_ids_state
                )

                F_matrix = self.F_state

            else:
                cluster_ids = None
                F_matrix = None

        # -------------------------------------------------
        # Output contract
        # -------------------------------------------------
        output = {
            # Learnable path; retains the autograd graph.
            "projected_features": (
                projected_features
            ),

            # Explicit structural state; no autograd graph.
            "H": H_state.detach(),

            # Current structural partition.
            "cluster_ids": (
                None
                if cluster_ids is None
                else cluster_ids.detach()
            ),

            "F_matrix": (
                None
                if F_matrix is None
                else F_matrix.detach()
            ),

            "structure_updated": bool(
                update_structure
            ),

            "structure_step": (
                self.structure_step
            ),

            "tensor_mode": tensor_mode,

            "backend_type": (
                self.structural_backend_type
            ),

            "scale_mode": (
                self.scale_mode
            ),
        }

        if return_diagnostics:
            output[
                "diagnostics"
            ] = {
                "initialization": (
                    initialization_diagnostics
                ),
                "kmeans": (
                    kmeans_diagnostics
                ),
                "refiner": (
                    refiner_diagnostics
                ),
            }

        return output

    # -----------------------------------------------------
    # Loss helper for the learnable projection W
    # -----------------------------------------------------

    @staticmethod
    def projection_reconstruction_loss(
        output,
        reduction="mean",
    ):
        """
        Compute the learnable reconstruction term between:

            projected_features = XW

        and the detached explicit state:

            H.

        Gradients flow into the projection/network path but
        not into H.
        """
        if not isinstance(
            output,
            dict,
        ):
            raise TypeError(
                "output must be the dictionary returned "
                "by top_agg_f.forward()."
            )

        if (
            "projected_features"
            not in output
            or "H" not in output
        ):
            raise KeyError(
                "output must contain "
                "'projected_features' and 'H'."
            )

        return F.mse_loss(
            output[
                "projected_features"
            ],
            output[
                "H"
            ].detach(),
            reduction=reduction,
        )

    # -----------------------------------------------------
    # State access
    # -----------------------------------------------------

    @torch.no_grad()
    def get_structural_state(
        self,
        clone=True,
    ):
        """
        Return the current explicit TAGSL state.
        """
        if not self.refiner.is_initialized:
            raise RuntimeError(
                "The structural state has not been "
                "initialized."
            )

        H_state = (
            self.refiner.get_H_state(
                clone=clone
            )
        )

        if self.is_cluster_state_initialized:
            if clone:
                cluster_ids = (
                    self.cluster_ids_state
                    .detach()
                    .clone()
                )

                F_matrix = (
                    self.F_state
                    .detach()
                    .clone()
                )

            else:
                cluster_ids = (
                    self.cluster_ids_state
                )

                F_matrix = self.F_state

        else:
            cluster_ids = None
            F_matrix = None

        return {
            "H": H_state,
            "cluster_ids": cluster_ids,
            "F_matrix": F_matrix,
            "structure_step": (
                self.structure_step
            ),
        }

    def extra_repr(
        self,
    ):
        return (
            f"num_nodes={self.num_nodes}, "
            f"input_dim={self.input_dim}, "
            f"embedding_dim={self.embedding_dim}, "
            f"num_clusters={self.num_clusters}, "
            f"alpha={self.alpha}, "
            f"beta={self.beta}, "
            f"landmark_size={self.landmark_size}, "
            f"backend="
            f"'{self.structural_backend_type}', "
            f"affinity_mode="
            f"'{self.affinity_mode}', "
            f"scale_mode="
            f"'{self.scale_mode}', "
            f"orthogonality_mode="
            f"'{self.orthogonality_mode}', "
            f"H_initialized="
            f"{self.is_H_initialized}, "
            f"structure_step="
            f"{self.structure_step}"
        )
        
        



# =========================================================
# TAGSL semantic alignment head
# =========================================================

class TAGSLSemanticAlignment(nn.Module):
    """
    TAGSL Phase-3 semantic alignment module.

    Given the explicit structural embedding H and the initial
    adjacency A, this module computes:

        1. Student-t soft assignments S;
        2. self-training target distribution T_tgt;
        3. adjacency consistency MSE(H H^T, A);
        4. assignment cross-entropy CE(T_tgt, S).

    The clustering objective is:

        L_cluster
            =
        MSE(H H^T, A)
        + CE(T_tgt, S).

    Optimization-state convention
    -----------------------------
    H:
        Explicit structural state produced by the spectral
        refiner. H is detached inside forward(), so this module
        does not modify H or construct a gradient path through
        the structural update.

    centroids:
        Learnable nn.Parameter with shape [c, d]. Gradients from
        the assignment cross-entropy update these centroids.

    T_tgt:
        Constructed from S.detach(), so it acts as a fixed
        self-training target during each Adam update.
    """

    def __init__(
        self,
        num_clusters,
        embedding_dim,
        eps=1e-12,
        dtype=torch.float32,
        device=None,
    ):
        super().__init__()

        num_clusters = int(
            num_clusters
        )

        embedding_dim = int(
            embedding_dim
        )

        eps = float(
            eps
        )

        if num_clusters <= 0:
            raise ValueError(
                "num_clusters must be positive."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive."
            )

        if (
            not torch.isfinite(
                torch.tensor(eps)
            ).item()
            or eps <= 0
        ):
            raise ValueError(
                "eps must be a finite positive scalar."
            )

        dtype_probe = torch.empty(
            (),
            dtype=dtype,
        )

        if dtype not in {
            torch.float32,
            torch.float64,
        }:
            raise TypeError(
                "dtype must be torch.float32 or "
                "torch.float64."
            )

        if not dtype_probe.is_floating_point():
            raise TypeError(
                "dtype must be floating point."
            )

        self.num_clusters = (
            num_clusters
        )

        self.embedding_dim = (
            embedding_dim
        )

        self.eps = eps

        self.centroids = nn.Parameter(
            torch.zeros(
                num_clusters,
                embedding_dim,
                dtype=dtype,
                device=device,
            )
        )

        self.register_buffer(
            "_centroids_initialized",
            torch.tensor(
                False,
                dtype=torch.bool,
                device=device,
            ),
        )

    # -----------------------------------------------------
    # Initialization state
    # -----------------------------------------------------

    @property
    def is_initialized(
        self,
    ):
        return bool(
            self._centroids_initialized
            .item()
        )

    def _require_initialized(
        self,
    ):
        if not self.is_initialized:
            raise RuntimeError(
                "The semantic centroids have not been "
                "initialized. Call initialize_centroids_() "
                "before forward()."
            )

    # -----------------------------------------------------
    # Input validation
    # -----------------------------------------------------

    def _validate_H(
        self,
        H,
    ):
        if not torch.is_tensor(H):
            raise TypeError(
                "H must be a torch.Tensor."
            )

        if H.ndim != 2:
            raise ValueError(
                "H must have shape [N, d]."
            )

        if H.shape[0] <= 0:
            raise ValueError(
                "H must contain at least one node."
            )

        if H.shape[1] != self.embedding_dim:
            raise ValueError(
                "H has an invalid embedding dimension. "
                f"Expected {self.embedding_dim}, "
                f"but received {H.shape[1]}."
            )

        if H.device != self.centroids.device:
            raise RuntimeError(
                "H and semantic centroids must be on "
                "the same device."
            )

        if H.dtype != self.centroids.dtype:
            raise RuntimeError(
                "H and semantic centroids must have "
                "the same dtype."
            )

        if not torch.isfinite(
            H.detach()
        ).all():
            raise ValueError(
                "H contains NaN or Inf."
            )

        # H is an explicit alternating-optimization state.
        return H.detach()

    def _prepare_cluster_ids(
        self,
        cluster_ids,
        num_nodes,
    ):
        if not torch.is_tensor(
            cluster_ids
        ):
            cluster_ids = torch.as_tensor(
                cluster_ids
            )

        if cluster_ids.ndim != 1:
            raise ValueError(
                "cluster_ids must have shape [N]."
            )

        if cluster_ids.shape[0] != num_nodes:
            raise ValueError(
                "cluster_ids and H have different "
                "numbers of nodes."
            )

        if cluster_ids.dtype == torch.bool:
            raise TypeError(
                "cluster_ids must use an integer dtype."
            )

        if cluster_ids.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError(
                "cluster_ids must use an integer dtype."
            )

        cluster_ids = cluster_ids.to(
            device=self.centroids.device,
            dtype=torch.long,
        )

        if torch.any(
            cluster_ids < 0
        ):
            raise ValueError(
                "cluster_ids contain a negative label."
            )

        if torch.any(
            cluster_ids
            >= self.num_clusters
        ):
            raise ValueError(
                "cluster_ids contain a label greater "
                "than or equal to num_clusters."
            )

        return cluster_ids

    def _validate_adjacency(
        self,
        A_raw,
        num_nodes,
    ):
        if not torch.is_tensor(
            A_raw
        ):
            raise TypeError(
                "A_raw must be a torch.Tensor."
            )

        if A_raw.ndim != 2:
            raise ValueError(
                "A_raw must have shape [N, N]."
            )

        expected_shape = (
            num_nodes,
            num_nodes,
        )

        if tuple(
            A_raw.shape
        ) != expected_shape:
            raise ValueError(
                "A_raw must have shape "
                f"{expected_shape}, but received "
                f"{tuple(A_raw.shape)}."
            )

        if A_raw.device != self.centroids.device:
            raise RuntimeError(
                "A_raw and semantic centroids must be "
                "on the same device."
            )

        if A_raw.dtype != self.centroids.dtype:
            raise RuntimeError(
                "A_raw and semantic centroids must "
                "have the same dtype."
            )

        if A_raw.layout == torch.sparse_coo:
            A_value = A_raw.coalesce()

            if not torch.isfinite(
                A_value.values()
            ).all():
                raise ValueError(
                    "A_raw contains NaN or Inf."
                )

            if torch.any(
                A_value.values() < 0
            ):
                raise ValueError(
                    "A_raw must contain nonnegative "
                    "edge weights."
                )

            return A_value

        if A_raw.layout == torch.strided:
            if not torch.isfinite(
                A_raw
            ).all():
                raise ValueError(
                    "A_raw contains NaN or Inf."
                )

            if torch.any(
                A_raw < 0
            ):
                raise ValueError(
                    "A_raw must contain nonnegative "
                    "edge weights."
                )

            return A_raw

        raise TypeError(
            "A_raw must be either a dense tensor or "
            "a sparse COO tensor."
        )

    # -----------------------------------------------------
    # K-means centroid initialization
    # -----------------------------------------------------

    @torch.no_grad()
    def initialize_centroids_(
        self,
        H,
        cluster_ids,
        overwrite=False,
        return_diagnostics=False,
    ):
        """
        Initialize centroid j as the mean embedding of the
        nodes assigned to cluster j:

            mu_j =
                1 / |C_j|
                sum_{i in C_j} H_i.

        Initialization is performed once by default. Pass
        overwrite=True only when an explicit reinitialization
        is intended.
        """
        if (
            self.is_initialized
            and not overwrite
        ):
            raise RuntimeError(
                "The semantic centroids are already "
                "initialized. Pass overwrite=True to "
                "replace them."
            )

        H_value = self._validate_H(
            H
        )

        cluster_ids = (
            self._prepare_cluster_ids(
                cluster_ids=cluster_ids,
                num_nodes=H_value.shape[0],
            )
        )

        cluster_sizes = torch.bincount(
            cluster_ids,
            minlength=self.num_clusters,
        )

        if torch.any(
            cluster_sizes == 0
        ):
            empty_clusters = torch.nonzero(
                cluster_sizes == 0,
                as_tuple=False,
            ).flatten()

            raise RuntimeError(
                "Cannot initialize semantic centroids "
                "because the partition contains empty "
                f"clusters: {empty_clusters.tolist()}."
            )

        centroid_sums = torch.zeros_like(
            self.centroids
        )

        centroid_sums.index_add_(
            0,
            cluster_ids,
            H_value,
        )

        centroid_means = (
            centroid_sums
            / cluster_sizes.to(
                dtype=H_value.dtype
            ).unsqueeze(1)
        )

        if not torch.isfinite(
            centroid_means
        ).all():
            raise FloatingPointError(
                "The initialized centroids contain "
                "NaN or Inf."
            )

        self.centroids.copy_(
            centroid_means
        )

        self._centroids_initialized.fill_(
            True
        )

        if not return_diagnostics:
            return self.centroids

        diagnostics = {
            "cluster_sizes": (
                cluster_sizes.detach()
            ),
            "centroid_norms": (
                self.centroids
                .norm(
                    dim=1
                )
                .detach()
            ),
            "initialized": True,
            "overwritten": bool(
                overwrite
            ),
        }

        return (
            self.centroids,
            diagnostics,
        )

    @torch.no_grad()
    def clear_centroids_(
        self,
    ):
        """
        Reset the learnable centroid values and initialization
        flag. This does not recreate the nn.Parameter.
        """
        self.centroids.zero_()

        self._centroids_initialized.fill_(
            False
        )

        return self

    @torch.no_grad()
    def get_centroids(
        self,
        clone=True,
    ):
        self._require_initialized()

        if clone:
            return (
                self.centroids
                .detach()
                .clone()
            )

        return self.centroids

    # -----------------------------------------------------
    # Semantic alignment forward
    # -----------------------------------------------------

    def forward(
        self,
        H,
        A_raw,
    ):
        """
        Compute the TAGSL semantic-alignment objective.

        Returns
        -------
        assignments:
            Student-t assignment matrix S with shape [N, c].
            It retains gradients with respect to centroids.

        target_distribution:
            Detached self-training target T_tgt.

        predicted_labels:
            argmax_j S_ij.

        adjacency_mse:
            MSE(H H^T, A).

        assignment_ce:
            CE(T_tgt, S).

        clustering_loss:
            adjacency_mse + assignment_ce.
        """
        self._require_initialized()

        H_value = self._validate_H(
            H
        )

        A_value = (
            self._validate_adjacency(
                A_raw=A_raw,
                num_nodes=H_value.shape[0],
            )
        )

        assignments = (
            tagsl_student_t_assignment(
                H=H_value,
                centroids=self.centroids,
                eps=self.eps,
            )
        )

        target_distribution = (
            tagsl_target_distribution(
                assignments.detach(),
                eps=self.eps,
            ).detach()
        )

        assignment_ce = (
            tagsl_soft_cross_entropy(
                target=target_distribution,
                prediction=assignments,
                eps=self.eps,
            )
        )

        adjacency_mse = (
            tagsl_sparse_adjacency_mse(
                H=H_value,
                A=A_value,
            )
        )

        clustering_loss = (
            adjacency_mse
            + assignment_ce
        )

        predicted_labels = (
            assignments.argmax(
                dim=1
            ).detach()
        )

        for name, value in {
            "assignments": assignments,
            "target_distribution": (
                target_distribution
            ),
            "assignment_ce": assignment_ce,
            "adjacency_mse": adjacency_mse,
            "clustering_loss": (
                clustering_loss
            ),
        }.items():
            if not torch.isfinite(
                value
            ).all():
                raise FloatingPointError(
                    f"{name} contains NaN or Inf."
                )

        return {
            "assignments": assignments,
            "target_distribution": (
                target_distribution
            ),
            "predicted_labels": (
                predicted_labels
            ),
            "adjacency_mse": (
                adjacency_mse
            ),
            "assignment_ce": (
                assignment_ce
            ),
            "clustering_loss": (
                clustering_loss
            ),
        }

    def extra_repr(
        self,
    ):
        return (
            f"num_clusters={self.num_clusters}, "
            f"embedding_dim={self.embedding_dim}, "
            f"eps={self.eps}, "
            f"initialized={self.is_initialized}"
        )
