import pickle

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.utils.data import DataLoader
from datasets import MultiSetSequence
from models import MultiSetNet
from torch.nn import init
import math

num_pixels=6
path_to_data= 'data/multiset.pickle'
max_set_len=30
num_subpixels = 64
MSEloss=True

no_epochs=100
batch_size = 4
lr = 0.001
reduction='min'
criterion = nn.MSELoss()


def get_mask(sizes, max_size):
    n = torch.arange(max_size).reshape(1, -1)
    m = sizes.unsqueeze(2)
    mask = n < m
    return mask


with open(path_to_data, 'rb') as f:
    mini_multiset = pickle.load(f)
    f.close()
df = pd.DataFrame.from_dict(mini_multiset, orient='index')
train_df, test_df = train_test_split(df, test_size=0.33, random_state=44, shuffle=True)

traindata = MultiSetSequence(dict=train_df.to_dict(orient='index'), num_pixels=round(num_pixels * 0.67),
                                  max_ccds=max_set_len, num_subpixels=num_subpixels)
traindata.set_targets(gal_type='lrg')



trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=False)

feature_extractor = nn.Sequential(
            nn.Linear(15, 32),
            nn.ReLU(inplace=False),
            nn.Linear(32, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.7, inplace=False),
            nn.Linear(128, 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, 16),
            nn.ReLU(inplace=False)
        )

class InvLinear(nn.Module):
    r"""Permutation invariant linear layer.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
        reduction: Permutation invariant operation that maps the input set into a single
            vector. Currently, the following are supported: mean, sum, max and min.
    """

    def __init__(self, in_features, out_features, bias=True, reduction='mean'):
        super(InvLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        assert reduction in ['mean', 'sum', 'max', 'min'], \
            '\'reduction\' should be \'mean\'/\'sum\'\'max\'/\'min\', got {}'.format(reduction)

        self.reduction = reduction

        self.beta = nn.Parameter(torch.Tensor(self.in_features,
                                              self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, self.out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        init.xavier_uniform_(self.beta)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.beta)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


    def forward(self, X, mask=None):
        r"""
        Maps the input set X = {x_1, ..., x_M} to a vector y of dimension out_features,
        through a permutation invariant linear transformation of the form:
            $y = \beta reduction(X) + bias$
        Inputs:
        X: N sets of SP subpixels, each of which has size at most M where each element has dimension in_features
           (tensor with shape (N, SP ,M, in_features))
        mask: binary mask to indicate which elements in X are valid (byte tensor
            with shape (N, M) or None); if None, all sets have the maximum size M.
            Default: ``None``.
        Outputs:
        Y: N vectors of dimension out_features (tensor with shape (N, SP, out_features))
        """

        N, SP, M, _ = X.shape
        device = X.device

        y = torch.zeros(N, SP, self.out_features).to(device)

        if mask is None:
            mask = torch.ones(N, SP, M).byte().to(device)

        if self.reduction == 'mean':
            sizes = mask.float().sum(dim=2).unsqueeze(2)
            Z = X * mask.unsqueeze(3).float()
            y = (Z.sum(dim=2) @ self.beta) / sizes

        elif self.reduction == 'sum':
            Z = X * mask.unsqueeze(3).float()
            y = Z.sum(dim=2) @ self.beta

        elif self.reduction == 'max':
            Z = X.clone()
            Z[~mask] = float('-Inf')
            y = Z.max(dim=2)[0] @ self.beta

        else:  # min
            Z = X.clone()
            Z[~mask] = float('Inf')
            y = Z.min(dim=2)[0] @ self.beta

        if self.bias is not None:
            y += self.bias

        return y


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, reduction={}'.format(
            self.in_features, self.out_features,
            self.bias is not None, self.reduction)

adder = InvLinear(16, 1, reduction=reduction, bias=True)

mlp = nn.Sequential(
            nn.Linear(64 + 2, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.7, inplace=False),
            nn.Linear(64, 32),
            nn.ReLU(inplace=False),
            nn.Linear(32, 1),
            nn.ReLU(inplace=False)
        )


model = MultiSetNet(n_features=traindata.num_features, reduction=reduction)

optimiser = optim.Adam(model.parameters(), lr=lr)


def batch_training_loop():
    for i, (X1, X2, labels, set_sizes) in enumerate(trainloader):
        # Extract inputs and associated labels from dataloader batch

        mask = get_mask(set_sizes, X1.shape[2])

        y = feature_extractor(X1)
        y = adder.forward(y, mask=mask)
        y = torch.cat((y, X2.unsqueeze(2)), dim=1)
        y = y.squeeze()
        y = mlp(y)

        print(labels, y)
        # Compute Loss
        loss = criterion(y, labels)


def normal_loop():
    def masked(sizes, max_size):
        return (torch.arange(max_size).reshape(1, -1) < sizes.reshape(-1, 1))

    for i, (X1, X2, labels, set_sizes) in enumerate(trainloader):

        # Extract inputs and associated labels from dataloader batch
        X1 = X1.squeeze()

        X2 = X2.reshape(-1, 1)

        mask = masked(set_sizes, X1.shape[1])
        # Predict outputs (forward pass)

        predictions = model(X1, X2, mask=mask)

        print(labels, predictions)


#normal_loop()

batch_training_loop()

print()