import torch

from deepset_layers import InvLinear
import torch.nn as nn


class SetNet(nn.Module):
    def __init__(self, n_features=5, n_output=3, reduction='sum'):
        super(SetNet, self).__init__()

        # Takes an Input Tensor and applies transformations to last layer --> features
        # Output of Feature Layer: Tensor with Max.CCDs elements, which can now be passed to Set Layer
        self.feature_extractor = nn.Sequential(
            nn.Linear(n_features, 7),
            nn.ReLU(inplace=True),
            nn.Linear(7, 6),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7, inplace=True),
            nn.Linear(6, 5),
            nn.ReLU(inplace=True),
            nn.Linear(5, n_output),
            nn.ReLU(inplace=True)
        )

        self.adder = InvLinear(3, 1, reduction=reduction, bias=True)

        # Invariant Layer Influenced by Code from DPernes, but adapted for the current regression task instead of CNN

    def forward(self, X, mask=None):
        y = self.feature_extractor(X)
        y = self.adder(y, mask=mask)
        return y


class MultiSetNet(nn.Module):
    def __init__(self, n_features=8, n_output=3, n_subpix = 64, reduction='sum'):
        super(MultiSetNet, self).__init__()

        # Takes an Input Tensor and applies transformations to last layer --> features
        # Output of Feature Layer: Tensor with Max.CCDs elements, which can now be passed to Set Layer
        self.feature_extractor = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(inplace=False),
            nn.Linear(32, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.7, inplace=False),
            nn.Linear(128, 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, 16),
            nn.ReLU(inplace=False)
        )
        self.adder = InvLinear(16, 1, reduction=reduction, bias=True)

        self.mlp = nn.Sequential(
            nn.Linear(n_subpix + 2, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.7, inplace=False),
            nn.Linear(64, 32),
            nn.ReLU(inplace=False),
            nn.Linear(32, 1),
            nn.ReLU(inplace=False)
        )


        # Invariant Layer Influenced by Code from DPernes, but adapted for the current regression task instead of CNN, + added a new dimension for subpix


    def forward(self, X1, X2, mask=None):
        y = self.feature_extractor(X1)
        y = self.adder(y, mask=mask)
        y = torch.cat((y, X2.unsqueeze(2)), dim=1).squeeze()
        y = self.mlp(y)
        return y


class VarMultiSetNet(nn.Module):
    def __init__(self, feature_extractor, mlp, med_layer, reduction):
        super(VarMultiSetNet, self).__init__()

        # Takes an Input Tensor and applies transformations to last layer --> features
        # Output of Feature Layer: Tensor with Max.CCDs elements, which can now be passed to Set Layer

        self.feature_extractor = feature_extractor
        self.adder = InvLinear(med_layer, 1, reduction=reduction, bias=True)

        self.mlp = mlp


        # Invariant Layer Influenced by Code from DPernes, but adapted for the current regression task instead of CNN, + added a new dimension for subpix


    def forward(self, X1, X2, mask=None):
        y = self.feature_extractor(X1)
        y = self.adder(y, mask=mask)
        y = torch.cat((y, X2.unsqueeze(2)), dim=1).squeeze()
        y = self.mlp(y)
        return y

