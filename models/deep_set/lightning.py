"""Building all necessary Lightning utilities, including Module and Datamodule"""

import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from util import get_mask, get_dataset, get_full_dataset


class LitVarDeepSet(pl.LightningModule):

    def __init__(self, model, lr, criterion):
        super().__init__()

        self.criterion = criterion
        self.lr = lr

        # Takes an Input Tensor and applies transformations to last layer --> features
        # Output of Feature Layer: Tensor with Max.CCDs elements, which can now be passed to Set Layer

        self.model = model
        self.save_hyperparameters()

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

    def test_step(self, batch, batch_idx):
        X1, X2, labels, set_sizes = batch

        mask = get_mask(set_sizes, X1.shape[2])
        # Predict outputs (forward pass)

        predictions = self(X1, X2, mask=mask)

        # Compute Loss
        test_loss = self.criterion(predictions, labels)

        self.log("Test_loss", test_loss, on_epoch=True, prog_bar=True)

        return test_loss


class DeepDataModule(pl.LightningDataModule):
    def __init__(self, area, num_pixels, gal, path_to_data, batch_size=128):
        super().__init__()
        self.batch_size = batch_size
        self.num_pixels = num_pixels
        self.area = area
        if self.area == "north":
            self.max_set_len = 30
        elif self.area == "south":
            self.max_set_len = 25
        else:
            self.max_set_len = 50

        self.gal = gal
        self.path_to_data = path_to_data

        self.traindata, self.valdata, self.testdata = get_full_dataset(area=self.area, num_pixels=self.num_pixels,
                                                                       max_set_len=self.max_set_len,
                                                                       gal=self.gal)
        self.num_features = self.traindata.num_features

    """
    def setup(self, stage: Optional[str] = None) -> None:
        """

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.traindata, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=8
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valdata, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=8
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.testdata, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=8
        )
