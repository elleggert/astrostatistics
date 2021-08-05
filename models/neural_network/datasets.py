from torch.utils.data import DataLoader, Dataset
import torch
import random
import pandas as pd
from sklearn import preprocessing
import numpy as np

class DensitySurvey(Dataset):
    def __init__(self, df, galaxy_type, scaler_in=None, scaler_out=None):
        self.data = df
        # Extracting Targets and Input
        self.target = self.data[galaxy_type].to_numpy(copy=True).reshape(-1, 1)
        self.input = self.data.drop(columns=['lrg','elg','qso']).to_numpy(copy=True)

        """
        # Scaling, when scaler is passed (test-set) use the existing scaler
        self.scaler_in = scaler_in
        self.scaler_out = scaler_out
        if self.scaler_in is None:
            self.scaler_in = preprocessing.MinMaxScaler()
            self.scaler_out = preprocessing.MinMaxScaler()
            self.input = self.scaler_in.fit_transform(self.input)
            self.target = self.scaler_out.fit_transform(self.target.reshape(-1, 1))
        else:
            self.input = self.scaler_in.transform(self.input)
            self.target = self.scaler_out.transform(self.target.reshape(-1, 1))

        """
    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return torch.from_numpy(self.input[idx]).float(), torch.tensor(self.target[idx]).float()

    """
    def __getscaler__(self):
        return self.scaler_in, self.scaler_out
    """