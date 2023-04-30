import torch
from torch import nn
from torch.optim import Adam
from torchmetrics import F1Score

from project.models import GCNEncoder, DotProductPredictor, DGCNEncoder, MLPPredictor


class LinkPredictor(nn.Module):
    def __init__(self, nfeats: int, nhids: int, learning_rate: float = 0.0001, device = 'cpu') -> None:
        super().__init__()

        self.learning_rate = learning_rate

        self.encoder = DGCNEncoder(nfeats, nhids)
        self.predictor = MLPPredictor(encoder=self.encoder, nfeats=nhids).to(device)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(self.predictor.parameters(), lr=self.learning_rate, weight_decay=5e-4)

        self.f1_score = F1Score(task='binary').to(device)
    

    def forward(self, x, edge_index):
        return self.predictor(edge_index=edge_index, x=x)

    def training_step(self, data):
        self.optimizer.zero_grad()
        labels = data.edge_label.float()
        
        logits = self(data.x, data.edge_index)
        
        loss = self.loss_fn(logits, labels)

        loss.backward()
        self.optimizer.step()

        return loss

    @torch.no_grad()
    def validation_step(self, data):
        labels = data.edge_label.float()

        logits = self(data.x, data.edge_index)
        y_pred = torch.sigmoid(logits)

        loss = self.loss_fn(logits, labels)

        f1_score = self.f1_score(y_pred, labels)

        return loss, f1_score
