from torch import nn
import torch.nn.functional as F


class LinearRegression(nn.Module):
    def __init__(self, n_input_vars=17, n_output_vars=1):
        super().__init__()  # call constructor of superclass
        self.linear = nn.Linear(n_input_vars, n_output_vars)

    def forward(self, x):
        return self.linear(x)


class BaseNet(nn.Module):
    def __init__(self, n_feature=16, mlp=None):
        super(BaseNet, self).__init__()

        # Takes an Input Tensor and applies transformations to last layer --> features
        # Output of Feature Layer: Tensor with Max.CCDs elements, which can now be passed to Set Layer

        if mlp is None:
            self.mlp = nn.Sequential(
                nn.Linear(n_feature, 32),
                nn.ReLU(inplace=False),
                nn.Linear(32, 128),
                nn.ReLU(inplace=False),
                nn.Dropout(p=0.7, inplace=False),
                nn.Linear(128, 64),
                nn.ReLU(inplace=False),
                nn.Linear(64, 16),
                nn.ReLU(inplace=False),
                nn.Linear(16, 1),
                nn.ReLU(inplace=False)
            )

        else:
            self.mlp = mlp

    def forward(self, X):
        out = self.mlp(X)
        return out
