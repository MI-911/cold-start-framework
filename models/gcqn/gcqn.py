from typing import Dict

from dgl import DGLGraph
from dgl.nn import GraphConv
import torch as tt
from torch import nn
from torch.nn import functional as ff
import numpy as np

import dgl.function as fn
import torch.optim as optim


def merge(embeddings):
    return tt.sigmoid(embeddings.sum(dim=0))


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
                name: nn.Linear(in_size, out_size) for name in etypes
            })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}


class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size, hidden_size, embedding_size):
        super(HeteroRGCN, self).__init__()
        # Use trainable node embeddings as featureless inputs.
        embed_dict = {node_type: nn.Parameter(tt.Tensor(G.number_of_nodes(node_type), in_size))
                      for node_type in G.ntypes}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)

        # Entity embedding layers
        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, embedding_size, G.etypes)

        # User embedding layer
        self.user_embedding_layer = nn.Linear(embedding_size * 3, embedding_size)
        self.loss_fnc = ff.smooth_l1_loss
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

        self.device = tt.device('cuda' if tt.cuda.is_available() else 'cpu')
        self.to(self.device)

    def get_loss(self, x, y):
        return self.loss_fnc(x, y)

    def get_user_embedding(self, liked_embedding, disliked_embedding, unknown_embedding):
        return tt.sigmoid(self.user_embedding_layer(tt.cat([liked_embedding, disliked_embedding, unknown_embedding])))

    def forward(self, G, liked_indices, unknown_indices, disliked_indices):
        h_dict = self.layer1(G, self.embed)
        h_dict = {k: ff.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        h_dict = {k: tt.sigmoid(h) for k, h in h_dict.items()}

        # Get entity embeddings
        entity_embeddings = h_dict['ENTITY']

        batch_size = liked_indices.shape[0]
        predictions = tt.zeros((batch_size, len(entity_embeddings)))

        for sample in range(batch_size):
            # Get the answered entity embeddings
            liked_embedding = merge(entity_embeddings[liked_indices[sample]])
            disliked_embedding = merge(entity_embeddings[disliked_indices[sample]])
            unknown_embedding = merge(entity_embeddings[unknown_indices[sample]])

            # Get the user embedding
            user_embedding = self.get_user_embedding(liked_embedding, disliked_embedding, unknown_embedding)

            # Get predictions
            prediction = entity_embeddings @ user_embedding
            predictions[sample] = prediction

        return predictions
