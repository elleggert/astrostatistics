"""Final Run for the best hyperparameter configurations of the Neural Network"""

import argparse
import math

import numpy as np
import torch
from sklearn import metrics
from torch import nn, optim
from torch.utils.data import DataLoader

from models import BaseNet
from util import get_full_dataset, get_final_dataset

device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
num_workers = 0 if device == 'cpu:0' else 8


def main():
    parser = argparse.ArgumentParser(description='Base-Network using Average Systematics - Final Run',
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
    model = define_generic_model(galaxy=gal, area=area).to(device)
    print(model)
    lr, weight_decay, batch_size = get_hparams(galaxy=gal, area=area)
    #criterion = nn.MSELoss()
    criterion = nn.PoissonNLLLoss()

    print(f"Learning Rate: {lr}, weight decay: {weight_decay}, batch_size: {batch_size}")
    print()
    print(f" Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print()
    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    drop_last = True if (len(valdata.input) > batch_size) else False
    no_epochs = 1000
    testloader = torch.utils.data.DataLoader(testdata, batch_size=128, shuffle=False)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers, drop_last=drop_last)

    valloader = torch.utils.data.DataLoader(valdata, batch_size=batch_size, shuffle=False, drop_last=drop_last)

    rmse, r2 = 0, 0

    for epoch in range(no_epochs):

        model.train()

        train_loop(criterion, model, optimiser, trainloader)

        model.eval()
        y_gold, y_pred = val_loop(model, valloader)

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

        print(
            f"Target: {len(y_gold)}, NaN: {np.isnan(y_gold).sum()}, Max: {np.max(y_gold)}, Min: {np.min(y_gold)}, "
            f"Mean: {np.mean(y_gold)}")
        print(
            f"Prediction: {len(y_pred)}, NaN: {np.isnan(y_pred).sum()}, Max: {np.max(y_pred)}, Min: {np.min(y_pred)}, "
            f"Mean: {np.mean(y_pred)}")

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


def val_loop(model, valloader):
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
    return y_gold, y_pred


def train_loop(criterion, model, optimiser, trainloader):
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


def parse_command_line_args(args):
    global gal, area, num_pixels, num_features, path_to_data, max_set_len, traindata, valdata, testdata
    num_pixels = args['num_pixels']
    gal = args['gal_type']
    area = args['area']
    traindata, valdata, testdata = get_final_dataset(num_pixels=num_pixels, area=area, gal=gal)
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
        if galaxy == 'lrg': # done
            return 0.0014261033969104992, 0.028416090227268063, 32
        elif galaxy == 'elg': # done
            return 0.0005483155562317579, 0.0030780942332826744, 256
        elif galaxy == 'qso':# done
            return 2.870873422671021e-05, 0.009684600148157946, 256
        elif galaxy == 'glbg':# done
            return 0.0012021727952055794, 0.0011311838264277674, 256
        else: # done
            return 7.378549719315322e-05, 0.041856805620734086, 256

    elif area == "south":
        if galaxy == 'lrg': # done
            return 1.4486327797188878e-05, 0.013624934186382366, 128
        elif galaxy == 'elg': # done
            return 5.083620673436264e-05, 0.08754366669226324, 128
        elif galaxy == 'qso': # done
            return 0.0012021727952055794, 0.00047458046979852696, 128
        elif galaxy == 'glbg': # done
            return 0.00011913397792380141, 0.0037406889381795955, 256
        else: # done
            return 0.0002706380625918258, 0.0010891782949972563, 256
    else:
        if galaxy == 'lrg': # done
            return 0.00034294857273689517, 0.0004828378731781737, 256
        elif galaxy == 'elg': # done
            return 0.0005319590184319263, 0.02003499265389086, 128
        elif galaxy == 'qso': # done
            return 7.274339370730868e-05, 0.008569901646051892, 256
        elif galaxy == 'glbg': # done
            return 0.00017847465975742678, 0.11091240407712032, 128
        else: # done
            return 3.15390633591755e-05, 0.00714665318674708, 256

def define_generic_model(area, galaxy):
    n_layers_mlp = 4
    out_features_mlp = 256
    p = 0.25

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


def define_model(area, galaxy):
    # defines and returns the best models from HP Tuning
    if area == "north":
        if galaxy == 'lrg':
            n_layers_mlp = 4
            out_features_mlp = 64
            p = 0.1

        elif galaxy == 'elg':
            n_layers_mlp = 4
            out_features_mlp = 64
            p = 0.1

        elif galaxy == 'qso':
            n_layers_mlp = 4
            out_features_mlp = 60
            p = 0.15

        elif galaxy == 'glbg':
            n_layers_mlp = 4
            out_features_mlp = 60
            p = 0.15

        else:
            n_layers_mlp = 4
            out_features_mlp = 50
            p = 0.2

    elif area == "south":

        if galaxy == 'lrg':
            n_layers_mlp = 4
            out_features_mlp = 50
            p = 0.15


        elif galaxy == 'elg':
            n_layers_mlp = 2
            out_features_mlp = 50
            p = 0.1

        elif galaxy == 'qso':
            n_layers_mlp = 2
            out_features_mlp = 45
            p = 0.05

        elif galaxy == 'glbg':
            n_layers_mlp = 4
            out_features_mlp = 60
            p = 0.2

        else:
            n_layers_mlp = 4
            out_features_mlp = 50
            p = 0.05

    else:

        if galaxy == 'lrg':
            n_layers_mlp = 2
            out_features_mlp = 60
            p = 0.05

        elif galaxy == 'elg':
            n_layers_mlp = 4
            out_features_mlp = 60
            p = 0.1

        elif galaxy == 'qso':
            n_layers_mlp = 2
            out_features_mlp = 60
            p = 0.05

        elif galaxy == 'glbg':
            n_layers_mlp = 4
            out_features_mlp = 60
            p = 0.1

        else:
            n_layers_mlp = 4
            out_features_mlp = 50
            p = 0.05

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
