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
            nn.Linear(7, 5),
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
            nn.Linear(7, 5),
            nn.ReLU(inplace=True),
            nn.Linear(5, n_output),
            nn.ReLU(inplace=True)
        )
        self.adder = InvLinear(3, 1, reduction=reduction, bias=True)

        self.mlp = nn.Sequential(
            nn.Linear(n_subpix + 2,32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16,8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
            nn.ReLU(inplace=True)
        )


        # Invariant Layer Influenced by Code from DPernes, but adapted for the current regression task instead of CNN


    # Make sure to pass another dimension featuring EBV & Stellar Density (maybe after transformation), concat with adder output and feed to NLP

    """ Code Example_
    def forward(self, image, data):
        x1 = self.cnn(image)
        x2 = data
        
        x = torch.cat((x1, x2), dim=1) # or dim = 0
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    """
    def forward(self, X1, X2, mask=None):
        y = self.feature_extractor(X1)
        y = self.adder(y, mask=mask)
        y = torch.cat((y, X2), dim=0)
        y = self.mlp(y.T)
        return y

""" TODO
1. Train Loop with Batching
2. Masking Procedure, need to mask out singular values on top of those that have no CCDs
3. How to actually feed real data into the system to see if it can learn
"""
