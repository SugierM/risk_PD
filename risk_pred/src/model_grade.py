import torch.nn as nn
import torch


class FlexibleScoringNet(nn.Module):
    def __init__(self, input_dim, layers_list, dropout_rate=0.2):
        super(FlexibleScoringNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        layers = []
        in_features = input_dim
        
        for h_size in layers_list:
            layers.append(nn.Linear(in_features, h_size))
            layers.append(nn.BatchNorm1d(h_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = h_size
        
        layers.append(nn.Linear(in_features, 1))
        
        self.model = nn.Sequential(*layers).to(self.device)
        
    def forward(self, x):
        return self.model(x)