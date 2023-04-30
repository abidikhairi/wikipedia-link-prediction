import torch
import torch.nn as nn


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


class MLPPredictor(nn.Module):
    def __init__(self, encoder: nn.Module, nfeats: int) -> None:
        super().__init__()

        self.encoder = encoder

        self.mlp = nn.Linear(2 * nfeats, 1, False)

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)


        src = edge_index[0, :]
        tgt = edge_index[1, :]

        h_src = h.index_select(0, src)
        h_tgt = h.index_select(0, tgt)

        h_e = torch.cat([h_src, h_tgt], dim=1)

        return self.mlp(h_e)
