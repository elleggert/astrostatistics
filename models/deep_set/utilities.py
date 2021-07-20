
###  Hyperparameters and Training Loops
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time

from sklearn import metrics
from sklearn.model_selection import train_test_split

from models import SetNet, MultiSetNet
from datasets import SetSequence, MultiSetSequence
from torch.utils.data import DataLoader
import torch.optim as optim

# Defining Loss
criterion = nn.MSELoss()

#Defining Hyperparemeters
no_epochs = 100  #very low, but computational power not sufficient for more iterations
batch = 4
multi_batch = 1
learning_rate = 0.001

#Using the Adam Method for Stochastic Optimisation
#optimiser = optim.Adam(model.parameters(), lr=learning_rate)

galaxy_types = ['lrg', 'elg', 'qso']
device = 'cpu'


def get_mask(sizes, max_size):
    return (torch.arange(max_size).reshape(1, -1).to(sizes.device) < sizes.reshape(-1, 1))


def train(num_pixels = 1000):
    traindata = SetSequence(num_pixels=num_pixels, var_set_len=True)
    for gal in galaxy_types:
        model = SetNet(n_features=traindata.num_features, reduction='sum').to(device)
        optimiser = optim.Adam(model.parameters(), lr=learning_rate)
        print("GALAXY TYPE: ", gal)
        print()
        traindata.set_targets(gal_type=gal)

        time_start = time.time()

        for epoch in range(no_epochs):
            loss_per_epoch = 0
            #loading the training data from trainset and shuffling for each epoch
            trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch, shuffle=True)

            for i, (X, labels, set_sizes) in enumerate(trainloader):
                #Put Model into train mode
                model.train()

                #Extract inputs and associated labels from dataloader batch
                X = X.to(device)

                labels = labels.to(device)
                set_sizes = set_sizes.to(device)

                mask = get_mask(set_sizes, X.shape[1])

                #Predict outputs (forward pass)

                predictions = model(X, mask=mask)

                #Compute Loss
                loss = criterion(predictions, labels)

                #Zero-out the gradients before backward pass (pytorch stores the gradients)
                optimiser.zero_grad()
                #Backpropagation
                loss.backward()
                #Perform one step of gradient descent
                optimiser.step()
                #Append loss to the general loss for this one epoch
                loss_per_epoch += loss.item()

            if epoch % 10 == 0:
                print("Loss for Epoch", epoch, ": ", loss_per_epoch)

        time_end = time.time()
        time_passed = time_end - time_start
        print()
        print(f"{time_passed / 60:.5} minutes ({time_passed:.3} seconds) taken to train the model")
        print()

def multi_train(num_pixels = 1000):

    traindata = MultiSetSequence(num_pixels=num_pixels)

    for gal in galaxy_types:
        model = MultiSetNet(n_features=traindata.num_features, reduction='sum').to(device)
        optimiser = optim.Adam(model.parameters(), lr=learning_rate)
        print("GALAXY TYPE: ", gal)
        print()
        traindata.set_targets(gal_type=gal)

        time_start = time.time()

        for epoch in range(no_epochs):
            loss_per_epoch = 0
            # loading the training data from trainset and shuffling for each epoch
            trainloader = torch.utils.data.DataLoader(traindata, batch_size=multi_batch, shuffle=True)

            for i, (X, labels, set_sizes) in enumerate(trainloader):

                model.train()

                # Extract inputs and associated labels from dataloader batch
                X = X.squeeze().to(device)

                labels = labels.to(device)

                # set_sizes = set_sizes.to(device)

                # mask = get_mask(set_sizes, X.shape[1])

                # Predict outputs (forward pass)

                # Not yet doing any masking
                predictions = model(X)

                # Compute Loss
                loss = criterion(predictions, labels)

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
        print(f"{time_passed / 60:.5} minutes ({time_passed:.3} seconds) taken to train the model")
        print()


class MultiSetTrainer:

    def __init__(self, num_pixels = 1000):
        #if traindata is None and testdata is None:
        with open('../../bricks_data/mini_multiset.pickle', 'rb') as f:
            mini_multiset = pickle.load(f)
            f.close()
        df = pd.DataFrame.from_dict(mini_multiset, orient='index')
        train_df, test_df = train_test_split(df, test_size=0.33, random_state=44, shuffle=True)

        print(f"Training Samples: {len(train_df)}")
        print(f"Test Samples: {len(test_df)}")

        self.traindata = MultiSetSequence(dict=train_df.to_dict(orient='index'), num_pixels=num_pixels)
        self.testdata = MultiSetSequence(dict=test_df.to_dict(orient='index'), num_pixels=num_pixels)

        self.models = []

        # Defining Loss
        self.criterion = nn.MSELoss()

        # Defining Hyperparemeters
        self.no_epochs = 100  # very low, but computational power not sufficient for more iterations
        self.multi_batch = 1
        self.learning_rate = 0.001

        # Using the Adam Method for Stochastic Optimisation
        # optimiser = optim.Adam(model.parameters(), lr=learning_rate)

        self.galaxy_types = ['lrg', 'elg', 'qso']
        self.device = 'cpu'
        self.reduction = 'sum'

    def train(self):

        for gal in self.galaxy_types:
            model = MultiSetNet(n_features=self.traindata.num_features, reduction=self.reduction).to(self.device)
            optimiser = optim.Adam(model.parameters(), lr=learning_rate)
            print("GALAXY TYPE: ", gal)
            print()
            self.traindata.set_targets(gal_type=gal)

            time_start = time.time()

            for epoch in range(self.no_epochs):
                loss_per_epoch = 0
                # loading the training data from trainset and shuffling for each epoch
                trainloader = torch.utils.data.DataLoader(self.traindata, batch_size=self.multi_batch, shuffle=True)

                for i, (X, labels, set_sizes) in enumerate(trainloader):
                    model.train()

                    # Extract inputs and associated labels from dataloader batch
                    X = X.squeeze().to(self.device)

                    labels = labels.to(self.device)

                    # set_sizes = set_sizes.to(device)

                    # mask = get_mask(set_sizes, X.shape[1])

                    # Predict outputs (forward pass)

                    # Not yet doing any masking
                    predictions = model(X)

                    # Compute Loss
                    loss = criterion(predictions, labels)

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
            print(f"{time_passed / 60:.5} minutes ({time_passed:.3} seconds) taken to train the model")
            print()
            self.models.append(model)


    def test(self):

        for i, gal in enumerate(galaxy_types):
            model = self.models[i]
            model.eval()
            y_pred = np.array([])
            testloader = torch.utils.data.DataLoader(self.testdata, batch_size=self.multi_batch, shuffle=False)

            for i, (X, labels, set_sizes) in enumerate(testloader):

                # Extract inputs and associated labels from dataloader batch
                X = X.squeeze().to(self.device)


                # Predict outputs (forward pass)
                # Not yet doing any masking
                predictions = model(X)

                # Get predictions and append to label array + count number of correct and total
                y_pred = np.append(y_pred, predictions.detach().numpy())

            y_gold = self.testdata.target

            print()
            print(f"MultiSetNet R^2 for {gal} :  {metrics.r2_score(y_gold, y_pred)}.")
            print(f"MultiSetNet MSE for {gal} :  {metrics.mean_squared_error(y_gold, y_pred)}.")
