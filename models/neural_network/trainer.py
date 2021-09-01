"""Trainer Utility for Neural Network. Called by main.py development function."""

import time

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from models import BaseNet
from datasets import DensitySurvey


class BaseNNTrainer:
    """ Class to Train and Test a given feed forward neural network

    Input: the hyperparameters to be trained on

    Output: Training of 3 models for 3 types and testing their performance on a test-set"""

    def __init__(self, num_pixels=None, kit=False, MSEloss=True, no_epochs=100, batch_size=1, lr=0.001):

        if kit:
            self.df = pd.read_csv('../../bricks_data/dataset_kitanidis.csv')
        else:
            self.df = pd.read_csv('../../bricks_data/dataset_geometric.csv')

        # ToDo: At later stage you can pass a list of pixel indeces to filter test and train sets

        if num_pixels is not None:
            self.df = self.df.sample(n=num_pixels, replace=False, random_state=44, axis=0)

        self.df.drop('pixel_id', axis=1, inplace=True)

        self.train_df, self.test_df = train_test_split(self.df, test_size=0.33, random_state=44, shuffle=True)

        self.models = []

        # Defining Loss
        if MSEloss:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.L1Loss()

            # Defining Hyperparemeters
        self.no_epochs = no_epochs  # very low, but computational power not sufficient for more iterations
        self.batch = batch_size
        self.learning_rate = lr
        self.galaxy_types = ['lrg', 'elg', 'qso']
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
        self.num_workers = 0 if self.device == 'cpu:0' else 8
        self.print_model_info()

    def print_model_info(self):
        print()
        print('++++++++ Model Characteristics +++++++')
        print()
        print(f"Training Samples: {len(self.train_df)}")
        print(f"Test Samples: {len(self.test_df)}")
        print(f"Loss: {self.criterion}")
        print(f"Training Epochs: {self.no_epochs}")
        print(f"Batch Size: {self.batch}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Device: {self.device}")
        print(f"Number of Workers: {self.num_workers}")

        print()
        print('+++++++++++++++++++++++++++++++++++++')

    def train_test(self):

        for gal in self.galaxy_types:
            model = BaseNet().to(self.device)
            print(f"Model {gal} params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}.")

            optimiser = optim.Adam(model.parameters(), lr=self.learning_rate)

            traindata = DensitySurvey(self.train_df, gal)
            scaler_in, scaler_out = traindata.__getscaler__()
            testdata = DensitySurvey(self.test_df, gal, scaler_in, scaler_out)
            print("Galaxy Type: ", str.upper(gal))
            print()

            time_start = time.time()

            trainloader = torch.utils.data.DataLoader(traindata, batch_size=self.batch, shuffle=True,
                                                      num_workers=self.num_workers, drop_last=True)

            for epoch in range(self.no_epochs):
                loss_per_epoch = 0

                for i, (inputs, labels) in enumerate(trainloader):
                    model.train()

                    # Extract inputs and associated labels from dataloader batch
                    inputs = inputs.to(self.device)

                    labels = labels.to(self.device)

                    # Predict outputs (forward pass)

                    predictions = model(inputs)

                    # Compute Loss
                    loss = self.criterion(predictions, labels)

                    # Zero-out the gradients before backward pass (pytorch stores the gradients)
                    optimiser.zero_grad()

                    # Backpropagation
                    loss.backward()

                    # Perform one step of gradient descent
                    optimiser.step()

                    # Append loss to the general loss for this one epoch
                    loss_per_epoch += loss.item()
                if epoch % 10 == 0:
                    print("Loss for Epoch", epoch, ": ", loss_per_epoch)
            time_end = time.time()
            time_passed = time_end - time_start
            print()
            print(f'{time_passed / 60:.5} minutes ({time_passed:.3} seconds) taken to train the model')
            print()
            self.models.append(model)

            model.eval()
            y_pred = np.array([])
            testloader = torch.utils.data.DataLoader(testdata, batch_size=self.batch, shuffle=False)

            for i, (inputs, labels) in enumerate(testloader):
                # Split dataloader
                inputs = inputs.to(self.device)
                # Forward pass through the trained network
                outputs = model(inputs)

                # Get predictions and append to label array + count number of correct and total
                y_pred = np.append(y_pred, outputs.detach().numpy())

            y_gold = testdata.target

            print()
            print(f"Neural Net R^2 for {gal} :  {metrics.r2_score(y_gold, y_pred)}.")
            print(f"Neural Net MSE for {gal} :  {metrics.mean_squared_error(y_gold, y_pred)}.")
