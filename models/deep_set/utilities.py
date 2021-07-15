
###  Hyperparameters and Training Loops
import torch
import torch.nn as nn
import time
from models import SetNet
from datasets import SetSequence
from torch.utils.data import DataLoader

import torch.optim as optim

# Defining Loss
criterion = nn.MSELoss()

#Defining Hyperparemeters
no_epochs = 100  #very low, but computational power not sufficient for more iterations
batch = 4
learning_rate = 0.001

#Using the Adam Method for Stochastic Optimisation
#optimiser = optim.Adam(model.parameters(), lr=learning_rate)

galaxy_types = ['lrg', 'elg', 'qso']
device = 'cpu'


def get_mask(sizes, max_size):
    return (torch.arange(max_size).reshape(1, -1).to(sizes.device) < sizes.reshape(-1, 1))


def train():
    traindata = SetSequence(num_pixels=1000, var_set_len=True)
    for gal in galaxy_types:
        model = SetNet(n_features=traindata.num_features, reduction='max').to(device)
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
