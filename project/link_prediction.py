import torch.nn as nn
from torch.optim import Adam
from torchmetrics import F1Score
from torch import binary_cross_entropy_with_logits

from project.models import GCNEncoder, DotProductPredictor


class LinkPredictor(nn.Module):
    def __init__(self, nfeats: int, nhids: int, learning_rate: float = 0.0001, device = 'cpu') -> None:
        super().__init__()

        self.learning_rate = learning_rate

        self.encoder = GCNEncoder(nfeats, nhids)
        self.predictor = DotProductPredictor(self.encoder).to(device)

        self.loss_fn = binary_cross_entropy_with_logits
        self.optimizer = Adam(self.predictor.parameters(), lr=self.learning_rate, weight_decay=5e-4)

        self.f1_score = F1Score(task='binary')
    

    def forward(self, x, edge_index):
        return self.predictor(edge_index=edge_index, x=x)

    def training_step(self, data):
        self.optimizer.zero_grad()
        
        logits = self(data.x, data.edge_index)
        loss = self.loss_fn(logits, data.edge_labels)
        
        import pdb; pdb.set_trace()

        return loss


    def validation_step(self, data):
        logits = self(data.x, data.edge_index)
        import pdb; pdb.set_trace()
        loss = self.loss_fn(logits, data.edge_labels)

        return loss