r"""
Permutation Equivariant and Permutation Invariant layers, as described in the
paper Deep Sets, by Zaheer et al. (https://arxiv.org/abs/1703.06114)
"""

""" Original Structure borrowed from https://github.com/dpernes/deepsets-digitsum.
Layer structure adapted to include another dimension in the input tensors, with masking now working on two dimensions.
"""

import math

import torch
from torch import nn
from torch.nn import init


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
        # print("INVLAYER:", X.shape)
        N, SP, M, _ = X.shape
        y = torch.zeros(N, SP, self.out_features)
        if mask is None:
            mask = torch.ones(N, SP, M).byte()

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
