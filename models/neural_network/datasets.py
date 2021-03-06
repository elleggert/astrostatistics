"""Dataset class for the Neural Networks Architecture"""

from torch.utils.data import DataLoader, Dataset
import torch
import random
import pandas as pd
from sklearn import preprocessing
import numpy as np


class DensitySurvey(Dataset):
    def __init__(self, df, galaxy_type):
        self.data = df
        # Extracting Targets and Input
        self.target = self.data[galaxy_type].to_numpy(copy=True).reshape(-1, 1)
        self.input = self.data.drop(columns=['lrg', 'elg', 'qso', 'glbg', 'rlbg']).to_numpy(copy=True)
        self.num_features = self.input.shape[1]

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return torch.from_numpy(self.input[idx]).float(), torch.tensor(self.target[idx]).float()
