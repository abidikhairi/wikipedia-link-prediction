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
    
class DGCNEncoder(nn.Module):
    def __init__(self, nfeats: int, nhids: int) -> None:
        super().__init__()

        self.linear = nn.Linear(nfeats, nhids)

        self.conv1 = GraphConv(in_channels=nfeats, out_channels=nhids)
        self.conv2 = GraphConv(in_channels=nhids, out_channels=nhids)

    def forward(self, x, edge_index):
        x_proj = self.linear(x)

        x = torch.dropout(self.conv1(x, edge_index), p=0.6, train=self.training)
        x = torch.relu(x)
        x = x + x_proj

        x = torch.dropout(self.conv2(x, edge_index), p=0.6, train=self.training)
        x = torch.relu(x)
        x = x + x_proj

        return x
