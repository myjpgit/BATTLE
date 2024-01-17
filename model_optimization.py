import copy
import math

import torch

from graph_learners import *
from layers import GCNConv_dense, GCNConv_dgl, SparseDropout
from torch.nn import Sequential, Linear, ReLU
from typing import Any, Dict, List, Optional, Union, Callable
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Identity, Linear, ModuleList

from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GCN for evaluation.
# class GCN(torch.nn.Module):
#     """The Graph Neural Network from the `"Semi-supervised
#     Classification with Graph Convolutional Networks"
#     <https://arxiv.org/abs/1609.02907>`_ paper, using the
#     :class:`~torch_geometric.nn.conv.GCNConv` operator for message passing.
#     Adapted from PyG for upward compatibility
#     Args:
#         in_channels (int): Size of each input sample.
#         hidden_channels (int): Size of each hidden sample.
#         num_layers (int): Number of message passing layers.
#         out_channels (int, optional): If not set to :obj:`None`, will apply a
#             final linear transformation to convert hidden node embeddings to
#             output size :obj:`out_channels`. (default: :obj:`None`)
#         dropout (float, optional): Dropout probability. (default: :obj:`0.`)
#         act (Callable, optional): The non-linear activation function to
#             use. (default: :obj:`"relu"`)
#         norm (torch.nn.Module, optional): The normalization operator to use.
#             (default: :obj:`None`)
#         jk (str, optional): The Jumping Knowledge mode
#             (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"lstm"`).
#             (default: :obj:`"last"`)
#         act_first (bool, optional): If set to :obj:`True`, activation is
#             applied before normalization. (default: :obj:`False`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch_geometric.nn.conv.GCNConv`.
#     """
#     def __init__(
#         self,
#         in_channels: int,
#         hidden_channels: int,
#         num_layers: int,
#         out_channels: Optional[int] = None,
#         dropout: float = 0.0,
#         act: Union[Callable, None] = F.relu,
#         norm: Optional[torch.nn.Module] = None,
#         jk: Optional[str] = None,
#         act_first: bool = False,
#         **kwargs,
#     ):
#         super().__init__()

#         self.in_channels = in_channels
#         self.hidden_channels = hidden_channels
#         self.num_layers = num_layers

#         self.dropout = dropout
#         self.act = act
#         self.jk_mode = jk
#         self.act_first = act_first

#         if out_channels is not None:
#             self.out_channels = out_channels
#         else:
#             self.out_channels = hidden_channels

#         self.convs = ModuleList()
#         self.convs.append(
#             self.init_conv(in_channels, hidden_channels, **kwargs))
#         for _ in range(num_layers - 2):
#             self.convs.append(
#                 self.init_conv(hidden_channels, hidden_channels, **kwargs))
#         if out_channels is not None and jk is None:
#             self._is_conv_to_out = True
#             self.convs.append(
#                 self.init_conv(hidden_channels, out_channels, **kwargs))
#         else:
#             self.convs.append(
#                 self.init_conv(hidden_channels, hidden_channels, **kwargs))

#         self.norms = None
#         if norm is not None:
#             self.norms = ModuleList()
#             for _ in range(num_layers - 1):
#                 self.norms.append(copy.deepcopy(norm))
#             if jk is not None:
#                 self.norms.append(copy.deepcopy(norm))

#         if jk is not None and jk != 'last':
#             self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

#         if jk is not None:
#             if jk == 'cat':
#                 in_channels = num_layers * hidden_channels
#             else:
#                 in_channels = hidden_channels
#             self.lin = Linear(in_channels, self.out_channels)

#     def init_conv(self, in_channels: int, out_channels: int,
#                   **kwargs) -> MessagePassing:
#         return GCNConv(in_channels, out_channels, **kwargs)

#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
#         for norm in self.norms or []:
#             norm.reset_parameters()
#         if hasattr(self, 'jk'):
#             self.jk.reset_parameters()
#         if hasattr(self, 'lin'):
#             self.lin.reset_parameters()

#     def forward(self, x: Tensor, edge_index: Adj, *args, **kwargs) -> Tensor:
#         """"""
#         xs: List[Tensor] = []
#         for i in range(self.num_layers):
#             x = self.convs[i](x, edge_index, *args, **kwargs)
#             if i == self.num_layers - 1 and self.jk_mode is None:
#                 break
#             if self.act_first:
#                 x = self.act(x)
#             if self.norms is not None:
#                 x = self.norms[i](x)
#             if not self.act_first:
#                 x = self.act(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#             if hasattr(self, 'jk'):
#                 xs.append(x)

#         x = self.jk(xs) if hasattr(self, 'jk') else x
#         x = self.lin(x) if hasattr(self, 'lin') else x
#         return x

#     def __repr__(self) -> str:
#         return (f'{self.__class__.__name__}({self.in_channels}, '
#                 f'{self.out_channels}, num_layers={self.num_layers})')

class GCN(nn.Module):
    # def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, Adj, sparse):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, adj, sparse):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()

        if sparse:
            self.layers.append(GCNConv_dgl(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv_dgl(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dgl(hidden_channels, out_channels))
        else:
            self.layers.append(GCNConv_dense(in_channels, hidden_channels))
            for i in range(num_layers - 2):
                self.layers.append(GCNConv_dense(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dense(hidden_channels, out_channels))

        self.dropout = dropout
        self.dropout_adj_p = dropout_adj
        self.Adj = adj
        self.Adj.requires_grad = False
        self.sparse = sparse

        if self.sparse:
            self.dropout_adj = SparseDropout(dprob=dropout_adj)
        else:
            self.dropout_adj = nn.Dropout(p=dropout_adj)

    def forward(self, x, Adj):

        if self.sparse:
            Adj = copy.deepcopy(Adj)
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(Adj)

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj)
        return x

class Encoder_Decoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, dropout, dropout_adj, adj, act, sparse):
        super(Encoder_Decoder, self).__init__()

        # split the number of layers for the encoder and decoders
        encoder_layers = int(num_layers / 2)
        decoder_layers = num_layers - encoder_layers

        self.shared_encoder = GCN(in_channels=in_dim, hidden_channels=hid_dim, out_channels=out_dim, num_layers=encoder_layers,
                                  dropout=dropout, dropout_adj=dropout_adj, adj=adj, sparse=sparse)

        self.attr_decoder = GCN(in_channels=out_dim, hidden_channels=hid_dim, out_channels=in_dim, num_layers=decoder_layers,
                                  dropout=dropout, dropout_adj=dropout_adj, adj=adj, sparse=sparse)

        self.struct_decoder = GCN(in_channels=out_dim, hidden_channels=hid_dim, out_channels=in_dim, num_layers=decoder_layers-1,
                                  dropout=dropout, dropout_adj=dropout_adj, adj=adj, sparse=sparse)


    def forward(self, x, Adj, anchor_adj):
        # encode
        h = self.shared_encoder(x, Adj)
        # decode feature matrix
        x_ = self.attr_decoder(h, anchor_adj)
        # decode adjacency matrix
        h_ = self.struct_decoder(h, anchor_adj)
        s_ = h_ @ h_.T
        # s_ = h @ h.T
        # return reconstructed matrices
        return x_, s_

class GraphEncoder(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse):

        super(GraphEncoder, self).__init__()
        self.dropout = dropout
        self.dropout_adj_p = dropout_adj
        self.sparse = sparse

        self.gnn_encoder_layers = nn.ModuleList()
        if sparse:
            self.gnn_encoder_layers.append(GCNConv_dgl(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, emb_dim))
        else:
            self.gnn_encoder_layers.append(GCNConv_dense(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, emb_dim))

        if self.sparse:
            self.dropout_adj = SparseDropout(dprob=dropout_adj)
        else:
            self.dropout_adj = nn.Dropout(p=dropout_adj)

        self.proj_head = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True),
                                           Linear(proj_dim, proj_dim))

    def forward(self,x, Adj_, branch=None):

        if self.sparse:
            if branch == 'anchor':
                Adj = copy.deepcopy(Adj_)
            else:
                Adj = Adj_
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(Adj_)

        for conv in self.gnn_encoder_layers[:-1]:
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gnn_encoder_layers[-1](x, Adj)
        z = self.proj_head(x)
        return z, x

class GCL(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse):
        super(GCL, self).__init__()

        self.encoder = GraphEncoder(nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse)

    def forward(self, x, Adj_, branch=None):
        z, embedding = self.encoder(x, Adj_, branch)
        return z, embedding

    # @staticmethod
    # def calc_loss(x, x_aug, temperature=0.2, sym=True):
    #     batch_size, _ = x.size()
    #     x_abs = x.norm(dim=1)
    #     x_aug_abs = x_aug.norm(dim=1)

    #     sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    #     sim_matrix = torch.exp(sim_matrix / temperature)
    #     pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    #     if sym:
    #         loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
    #         loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

    #         loss_0 = - torch.log(loss_0).mean()
    #         loss_1 = - torch.log(loss_1).mean()
    #         loss = (loss_0 + loss_1) / 2.0
    #         return loss
    #     else:
    #         loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    #         loss_1 = - torch.log(loss_1).mean()
    #         return loss_1

    @staticmethod
    def calc_loss(args, x, x_aug, x_local, temperature=0.2, sym=True):  # Calculating contrastive loss
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)  # anchor graph embedding 求指定维度上的范数
        x_local_abs = x_local.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)  # learned graph embedding

        sim_matrix_1 = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix_1 = torch.exp(sim_matrix_1 / temperature)
        pos_sim_1 = sim_matrix_1[range(batch_size), range(batch_size)]

        sim_matrix_2 = torch.einsum('ik,jk->ij', x_local, x_aug) / torch.einsum('i,j->ij', x_local_abs, x_aug_abs)
        sim_matrix_2 = torch.exp(sim_matrix_2 / temperature)
        pos_sim_2 = sim_matrix_1[range(batch_size), range(batch_size)]

        if sym:
            loss_0 = pos_sim_1 / (sim_matrix_1.sum(dim=0) - pos_sim_1)
            loss_1 = pos_sim_1 / (sim_matrix_1.sum(dim=1) - pos_sim_1)

            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()

            loss_2 = pos_sim_2 / (sim_matrix_1.sum(dim=0) - pos_sim_2)
            loss_3 = pos_sim_2 / (sim_matrix_1.sum(dim=1) - pos_sim_2)

            loss_2 = - torch.log(loss_2).mean()
            loss_3 = - torch.log(loss_3).mean()

            loss_aug = (loss_0 + loss_1) / 2.0
            loss_edge = (loss_2 + loss_3) / 2.0

            loss = args.loss_alpha * loss_aug + (1 - args.loss_alpha) * loss_edge
            return loss
        else:
            loss_1 = pos_sim_1 / (sim_matrix_1.sum(dim=1) - pos_sim_1)
            loss_1 = - torch.log(loss_1).mean()

            loss_2 = pos_sim_2 / (sim_matrix_2.sum(dim=1) - pos_sim_2)
            loss_2 = - torch.log(loss_2).mean()

            loss = args.loss_alpha * loss_1 + (1 - args.loss_alpha) * loss_2

            return loss