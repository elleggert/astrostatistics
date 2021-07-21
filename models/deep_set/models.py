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
    def __init__(self, n_features=9, n_output=3, n_subpix = 64, reduction='sum'):
        super(MultiSetNet, self).__init__()

        # Takes an Input Tensor and applies transformations to last layer --> features
        # Output of Feature Layer: Tensor with Max.CCDs elements, which can now be passed to Set Layer
        self.feature_extractor = nn.Sequential(
            nn.Linear(n_features, 7),
            nn.ReLU(inplace=True),
            nn.Linear(7, 6),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7, inplace=True),
            nn.Linear(6, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, n_output),
            nn.ReLU(inplace=True)
        )
        self.adder = InvLinear(n_output, 1, reduction=reduction, bias=True)

        self.mlp = nn.Sequential(
            nn.Linear(n_subpix + 2,32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7, inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
            nn.ReLU(inplace=True)
        )


        # Invariant Layer Influenced by Code from DPernes, but adapted for the current regression task instead of CNN


    # Make sure to pass another dimension featuring EBV & Stellar Density (maybe after transformation), concat with adder output and feed to NLP

    def forward(self, X1, X2, mask=None):
        y = self.feature_extractor(X1)
        y = self.adder(y, mask=mask)
        y = torch.cat((y, X2), dim=0)
        y = self.mlp(y.T)
        return y

