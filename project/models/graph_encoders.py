import torch
import torch.nn as nn

from torch_geometric.nn import GraphConv


class GCNEncoder(nn.Module):
    def __init__(self, nfeats: int, nhids: int) -> None:
        super().__init__()

        self.conv1 = GraphConv(in_channels=nfeats, out_channels=nhids)
        self.conv2 = GraphConv(in_channels=nhids, out_channels=nhids)

    def forward(self, x, edge_index):
        x = torch.dropout(self.conv1(x, edge_index), p=0.6, train=self.training)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)

        return x
    

class DotProductPredictor(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()

        self.encoder = encoder

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)

        src = edge_index[0, :]
        tgt = edge_index[1, :]

        h_src = h.index_select(0, src)
        h_tgt = h.index_select(0, tgt)

        return torch.bmm(h_src.unsqueeze(1), h_tgt.unsqueeze(2)).squeeze()
