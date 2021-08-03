import os
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from deepset_layers import InvLinear
from util import get_mask, get_dataset


class LitVarDeepSet(pl.LightningModule):

    def __init__(self, model, lr, criterion):
        super().__init__()

        self.criterion = criterion
        self.lr = lr

        # Takes an Input Tensor and applies transformations to last layer --> features
        # Output of Feature Layer: Tensor with Max.CCDs elements, which can now be passed to Set Layer

        self.model = model

    def forward(self, X1, X2, mask=None):
        return self.model(X1, X2, mask)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        X1, X2, labels, set_sizes = batch

        mask = get_mask(set_sizes, X1.shape[2])
        # Predict outputs (forward pass)

        predictions = self(X1, X2, mask=mask)

        # Compute Loss
        loss = self.criterion(predictions, labels)

        self.log("Train_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X1, X2, labels, set_sizes = batch

        mask = get_mask(set_sizes, X1.shape[2])
        # Predict outputs (forward pass)

        predictions = self(X1, X2, mask=mask)

        # Compute Loss
        val_loss = self.criterion(predictions, labels)

        self.log("Val_loss", val_loss, on_epoch=True, prog_bar=True)


        return val_loss

    def validation_epoch_end(self, validation_step_outputs):
        self.val_dataloader()


class DeepDataModule(pl.LightningDataModule):
    def __init__(self, num_pixels, max_set_len, gal, path_to_data, batch_size = 128):
        super().__init__()
        self.batch_size = batch_size
        self.num_pixels = num_pixels
        self.max_set_len = max_set_len
        self.gal = gal
        self.path_to_data = path_to_data

    def setup(self, stage: Optional[str] = None) -> None:
        self.traindata, self.valdata = get_dataset(num_pixels=self.num_pixels, max_set_len=self.max_set_len, gal=self.gal,
                                                   path_to_data=self.path_to_data)


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.traindata, batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=False
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valdata, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )



