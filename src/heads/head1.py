import torch.nn as nn
from headfactory import HeadFactory


@HeadFactory.register("head1")
class GlobalPoolLinearHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, dropout: float):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.mean(dim=1)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
