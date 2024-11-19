"""
A GNN used at test time that supports early stopping during the integrator
"""
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import argparse
from torch_geometric.nn import GCNConv, ChebConv  # noqa
# from run_GNN import get_optimizer, train, test
from early_stop_solver import EarlyStopInt
from base_classes import BaseGNN
from block_constant import ConstantODEblock
from function_laplacian_diffusion import LaplacianODEFunc
# from function_laplacian_diffusion_GCN import LaplacianODEFunc


class GNNEarly(BaseGNN):
    def __init__(self, opt, dataset, device=torch.device('cpu')):
        super(GNNEarly, self).__init__(opt, dataset, device)
        self.f = LaplacianODEFunc
        block = ConstantODEblock
        self.device = device
        self.data = dataset.data
        self.beta_train = torch.sigmoid(nn.Parameter(torch.tensor(0.0)))
        time_tensor = torch.tensor([0, self.T]).to(device)

        self.odeblock = block(self.f, self.regularization_fns, opt, dataset.data, device, t=time_tensor).to(device)
        # self.recover_edge()
        # overwrite the test integrator with this custom one
        with torch.no_grad():
            self.odeblock.test_integrator = EarlyStopInt(self.T, self.opt, self.device)
            self.set_solver_data(dataset.data)

    def set_solver_m2(self):
        self.odeblock.test_integrator.m2_weight = self.m2.weight.data.detach().clone().to(self.device)
        self.odeblock.test_integrator.m2_bias = self.m2.bias.data.detach().clone().to(self.device)

    def set_solver_data(self, data):
        self.odeblock.test_integrator.data = data

    def cleanup(self):
        del self.odeblock.test_integrator.m2
        torch.cuda.empty_cache()

    def forward(self, x):
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)

        self.odeblock.set_x0(x)

        with torch.no_grad():
            self.set_solver_m2()
      
        return z

    def forward_encoder(self, x, pos_encoding):
        if self.opt['use_labels']:
            y = x[:, -self.num_classes:]
            x = x[:, :-self.num_classes]

        if self.opt['beltrami']:
            x = self.mx(x)
            p = self.mp(pos_encoding)
            x = torch.cat([x, p], dim=1)
        else:
            x = self.m1(x)

        if self.opt['use_mlp']:
            x = F.dropout(x, self.opt['dropout'], training=self.training)
            x = x + self.m11(F.relu(x))
            x = x + self.m12(F.relu(x))

        if self.opt['use_labels']:
            x = torch.cat([x, y], dim=-1)

        if self.opt['batch_norm']:
            x = self.bn_in(x)

        # Solve the initial value problem of the ODE.
        if self.opt['augment']:
            c_aux = torch.zeros(x.shape).to(self.device)
            x = torch.cat([x, c_aux], dim=1)

        return x

    def forward_ODE(self, x, pos_encoding):
        x = self.forward_encoder(x, pos_encoding)

        self.odeblock.set_x0(x)

        if self.training and self.odeblock.nreg > 0:
            z, self.reg_states = self.odeblock(x)
        else:
            z = self.odeblock(x)

        if self.opt['augment']:
            z = torch.split(z, x.shape[1] // 2, dim=1)[0]

        return z
