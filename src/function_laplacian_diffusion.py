import torch
from torch import nn
import torch_sparse
import numpy as np

from base_classes import ODEFunc
from utils import MaxNFEException
from torch.nn.init import uniform
from torch_geometric.utils import get_laplacian

# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class LaplacianODEFunc(ODEFunc):

    # currently requires in_features = out_features
    def __init__(self, in_features, out_features, opt, data, device):
        super(LaplacianODEFunc, self).__init__(opt, data, device)

        self.data = data
        self.in_features = in_features
        self.out_features = out_features
        self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
        self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
        self.alpha_sc = nn.Parameter(torch.ones(1))
        # self.theta = torch.tensor(0.2 * (1-torch.exp(-self.theta_train)), requires_grad=True)
        # self.b_W = nn.Parameter(torch.Tensor(in_features))
        # self.reset_parameters()

    def reset_parameters(self):
        uniform(self.b_W, a=-1, b=1)

    def set_Beta(self, T=None):
        Beta = torch.diag(self.b_W)
        return Beta

    def sparse_multiply(self, x):
        if self.opt['block'] in ['attention']:  # adj is a multihead attention
            mean_attention = self.attention_weights.mean(dim=1)
            edge_p = (mean_attention - 0.0)
            ax = torch_sparse.spmm(self.edge_index, edge_p, x.shape[0], x.shape[0], x)
            ar = torch_sparse.spmm(self.edge_index, self.data.edge_weight, x.shape[0], x.shape[0], x)
        elif self.opt['block'] in ['mixed', 'hard_attention']:  # adj is a torch sparse matrix
            edge_p = (self.attention_weights - 0.0)
            ax = torch_sparse.spmm(self.edge_index, edge_p, x.shape[0], x.shape[0], x)
            ar = torch_sparse.spmm(self.edge_index, self.data.edge_weight, x.shape[0], x.shape[0], x)
        else:
            edge_p = (self.edge_weight - 0.4)  # cornell=0.4
            # edge_p = self.edge_weight - 0.5*torch.bernoulli(self.probs)
            ax = torch_sparse.spmm(self.edge_index, edge_p, x.shape[0], x.shape[0], x)
            ar = torch_sparse.spmm(self.edge_index, self.data.edge_weight, x.shape[0], x.shape[0], x)
        return ax, ar

    def forward(self, t, x):  # the t param is needed by the ODE solver.
        if self.nfe > self.opt["max_nfe"]:
            raise MaxNFEException
        self.nfe += 1
        # print(t)
        ax, ar = self.sparse_multiply(x)
        if not self.opt['no_alpha_sigmoid']:
            alpha = torch.sigmoid(self.alpha_train)
            delta_a = torch.sigmoid(self.delta_a)
        else:
            alpha = self.alpha_train
            delta_a = self.delta_a
        v_att = ax - x
        v_rep = ((ax - x) * v_att) / (self.data.degrees * (torch.norm(ax - x, dim=1))).unsqueeze(1)
        v = alpha * v_att - delta_a * v_rep
        if self.opt['add_source']:
            v = v + self.beta_train * self.x0
        return v
