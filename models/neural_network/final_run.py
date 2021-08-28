import argparse
import math

import numpy as np
import torch
from sklearn import metrics
from torch import nn, optim
from torch.utils.data import DataLoader

from models import BaseNet
from util import get_full_dataset

device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
num_workers = 0 if device == 'cpu:0' else 8


def main():
    parser = argparse.ArgumentParser(description='MBase-Network using Average Systematics - Final Run',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--num_pixels', default=None, metavar='', type=int, help='number of training examples')
    parser.add_argument('-a', '--area', default='des', metavar='', type=str,
                        help='The area of the sky that should be trained on')
    parser.add_argument('-g', '--gal_type', default='lrg', metavar='', type=str,
                        help='Galaxy Type to optimise model for')

    args = vars(parser.parse_args())

    parse_command_line_args(args)

    print_session_stats(args)

    best_r2 = -10000
    patience = 0
    model = define_model(galaxy=gal, area=area).to(device)
    print(model)
    lr, weight_decay, batch_size = get_hparams(galaxy=gal, area=area)
    print()
    print(f" Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print()
    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    drop_last = True if (len(valdata.input) > batch_size) else False
    no_epochs = 1000
    testloader = torch.utils.data.DataLoader(testdata, batch_size=128, shuffle=False)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers, drop_last=drop_last)

    valloader = torch.utils.data.DataLoader(valdata, batch_size=batch_size, shuffle=False, drop_last=drop_last)

    rmse, r2 = 0, 0

    for epoch in range(no_epochs):

        model.train()

        for i, (inputs, labels) in enumerate(trainloader):
            model.train()

            # Extract inputs and associated labels from dataloader batch
            inputs = inputs.to(device)

            labels = labels.to(device)

            # Predict outputs (forward pass)

            predictions = model(inputs)

            # Compute Loss
            loss = criterion(predictions, labels)

            # Zero-out the gradients before backward pass (pytorch stores the gradients)
            optimiser.zero_grad()

            # Backpropagation
            loss.backward()

            # Perform one step of gradient descent
            optimiser.step()

        model.eval()
        y_pred = np.array([])
        y_gold = np.array([])

        with torch.no_grad():

            for i, (inputs, labels) in enumerate(valloader):
                # Split dataloader
                inputs = inputs.to(device)
                # Forward pass through the trained network
                outputs = model(inputs)

                # Get predictions and append to label array + count number of correct and total
                y_pred = np.append(y_pred, outputs.cpu().detach().numpy())
                y_gold = np.append(y_gold, labels.cpu().detach().numpy())

        try:
            r2 = metrics.r2_score(y_gold, y_pred)
            rmse = math.sqrt(metrics.mean_squared_error(y_gold, y_pred))
            print("epoch", epoch, r2, rmse, patience)

            if r2 > best_r2:
                best_r2 = r2
                patience = 0
            else:
                patience += 1

            if patience > 7:
                break
        except:
            print("++++++++++++++++++++")
            print("        NaN         ")
            print("++++++++++++++++++++")

    model.eval()
    y_pred = np.array([])
    y_gold = np.array([])

    with torch.no_grad():

        for i, (inputs, labels) in enumerate(testloader):
            # Split dataloader
            inputs = inputs.to(device)
            # Forward pass through the trained network
            outputs = model(inputs)

            # Get predictions and append to label array + count number of correct and total
            y_pred = np.append(y_pred, outputs.cpu().detach().numpy())
            y_gold = np.append(y_gold, labels.cpu().detach().numpy())

        print("Target", len(y_gold), np.isnan(y_gold).sum(), np.max(y_gold), np.min(y_gold), np.mean(y_gold))
        print(y_gold)
        print("Prediction", len(y_pred), np.isnan(y_pred).sum(), np.max(y_pred), np.min(y_pred), np.mean(y_pred))
        print(y_pred)

        r2, rmse, mae = 0, 0, 0

        try:
            r2 = metrics.r2_score(y_gold, y_pred)
            rmse = math.sqrt(metrics.mean_squared_error(y_gold, y_pred))
            mae = metrics.mean_absolute_error(y_gold, y_pred)

        except:
            print("++++++++++++++++++++")
            print("   NaN Predicted    ")
            print("++++++++++++++++++++")

        print()
        print(f" XXXXXX======== TRIAL {area} - {gal} ended")
        print()
        print("Test Set - R-squared: ", r2)
        print("Test Set - RMSE: ", rmse)
        print("Test Set - MAE: ", mae)
        print()
        print()
        print()

    torch.save(model, f"trained_models/{area}/{gal}/{r2}.pt")


def parse_command_line_args(args):
    global gal, area, num_pixels, num_features, path_to_data, max_set_len, traindata, valdata, testdata
    num_pixels = args['num_pixels']
    gal = args['gal_type']
    area = args['area']
    traindata, valdata, testdata = get_full_dataset(num_pixels=num_pixels, area=area, gal=gal)
    num_features = traindata.num_features


def print_session_stats(args):
    print('++++++++ Session Characteristics +++++++')
    print()
    print(f"Area: {area}")
    print(f"Gal Type: {gal}")
    print(f"Training Samples: {len(traindata)}")
    print(f"Validation Samples: {len(valdata)}")
    print(f"Test Samples: {len(testdata)}")
    print(f"Number of features: {num_features}")
    print(f"Device: {device}")
    print(f"Number of Workers: {num_workers}")
    print()
    print('+++++++++++++++++++++++++++++++++++++++')

def get_hparams(galaxy, area):
    # defines and returns: lr, weight_decay, batch_size
    if area == "north":
        if galaxy == 'lrg':
            #return 0.1, 0.11966102805969332, 256
            return 0.00042966343711901, 0.0, 128
        elif galaxy == 'elg':
            return 0.0022168493798361945, 0,  16
        else:
            return 0.0059739578840763, 0.0, 16

    elif area == "south":
        if galaxy == 'lrg':
            return 4.8538834002443876e-05, 0, 128
        elif galaxy == 'elg':
            return 0.0007693503935423424, 0, 32
        else:
            return 0.0011912772207786039, 15312074922163604, 32
    else:
        if galaxy == 'lrg':
            return 0.0005231431812476474, 0, 32
        elif galaxy == 'elg':
            return  0.003155992400443771, 0, 128
        else:
            return  0.00011377624891759982, 0, 32

def define_model(area, galaxy):
    # defines and returns the best models from HP Tuning
    if area == "north":
        if galaxy == 'lrg':
            n_layers_mlp = 4
            out_features_mlp = 50
            p = 0.2

        elif galaxy == 'elg':
            n_layers_mlp = 2
            out_features_mlp = 50
            p = 0.2

        else:
            n_layers_mlp = 2
            out_features_mlp = 45
            p = 0.125

    elif area == "south":

        if galaxy == 'lrg':
            n_layers_mlp = 4
            out_features_mlp = 50
            p = .2


        elif galaxy == 'elg':
            n_layers_mlp = 4
            out_features_mlp = 40
            p = 0.15

        else:
            n_layers_mlp = 4
            out_features_mlp = 35
            p = 0.25

    else:

        if galaxy == 'lrg':
            n_layers_mlp = 4
            out_features_mlp = 60
            p = 0.2

        elif galaxy == 'elg':
            n_layers_mlp = 4
            out_features_mlp = 50
            p = 0.2

        else:
            n_layers_mlp = 4
            out_features_mlp = 60
            p = 0.15


    mlp_layers = []

    in_features = num_features

    for i in range(n_layers_mlp):
        mlp_layers.append(nn.Linear(in_features, out_features_mlp))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(p))
        in_features = out_features_mlp
    mlp_layers.append(nn.Linear(in_features, int(in_features / 2)))
    mlp_layers.append(nn.Linear(int(in_features / 2), 1))


    return BaseNet(mlp=nn.Sequential(*mlp_layers))


if __name__ == "__main__":
    main()
