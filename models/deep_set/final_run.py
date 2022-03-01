"""File to train the best models from hyperparameter tuning on the full trainingset"""

import argparse
import math

import numpy as np
import torch
from sklearn import metrics
from torch import nn, optim
from torch.utils.data import DataLoader

from models import VarMultiSetNet
from util import get_mask, get_full_dataset, get_final_dataset

device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
num_workers = 0 if device == 'cpu:0' else 8


def main():
    parser = argparse.ArgumentParser(description='MultiSetSequence DeepSet-Network - Full Run',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--num_pixels', default=None, metavar='', type=int, help='number of training examples')
    parser.add_argument('-a', '--area', default='north', metavar='', type=str,
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
    #criterion = nn.MSELoss()
    criterion = nn.PoissonNLLLoss()
    print(f"Learning Rate: {lr}, weight decay: {weight_decay}, batch_size: {batch_size}")
    print()
    print(f" Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print()
    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    drop_last = True if (len(valdata.input) > batch_size) else False
    no_epochs = 1000

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers, drop_last=drop_last)

    valloader = torch.utils.data.DataLoader(valdata, batch_size=batch_size, shuffle=False, drop_last=False)

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
            break

    testloader = torch.utils.data.DataLoader(testdata, batch_size=128, shuffle=False)

    model.eval()

    y_pred = np.array([])
    y_gold = np.array([])

    with torch.no_grad():

        for i, (X1, X2, labels, set_sizes) in enumerate(testloader):
            # Extract inputs and associated labels from dataloader batch
            X1 = X1.to(device)

            X2 = X2.to(device)

            labels = labels.to(device)

            set_sizes = set_sizes.to(device)

            mask = get_mask(set_sizes, X1.shape[2])
            # Predict outputs (forward pass)

            outputs = model(X1, X2, mask=mask)
            # Predict outputs (forward pass)
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
        for i, (X1, X2, labels, set_sizes) in enumerate(valloader):
            # Extract inputs and associated labels from dataloader batch
            X1 = X1.to(device)

            X2 = X2.to(device)

            labels = labels.to(device)

            set_sizes = set_sizes.to(device)

            mask = get_mask(set_sizes, X1.shape[2])
            # Predict outputs (forward pass)

            predictions = model(X1, X2, mask=mask)
            # Predict outputs (forward pass)

            # Get predictions and append to label array + count number of correct and total
            y_pred = np.append(y_pred, predictions.cpu().detach().numpy())
            y_gold = np.append(y_gold, labels.cpu().detach().numpy())
    return y_gold, y_pred


def train_loop(criterion, model, optimiser, trainloader):
    for i, (X1, X2, labels, set_sizes) in enumerate(trainloader):
        # Extract inputs and associated labels from dataloader batch
        X1 = X1.to(device)

        X2 = X2.to(device)

        labels = labels.to(device)

        set_sizes = set_sizes.to(device)

        mask = get_mask(set_sizes, X1.shape[2])
        # Predict outputs (forward pass)

        predictions = model(X1, X2, mask=mask)

        # Zero-out the gradients before backward pass (pytorch stores the gradients)

        optimiser.zero_grad()

        # Compute Loss
        loss = criterion(predictions, labels)

        # Backpropagation
        loss.backward()

        # Perform one step of gradient descent
        optimiser.step()


def parse_command_line_args(args):
    global gal, area, num_pixels, max_set_len, traindata, valdata, testdata, features
    num_pixels = args['num_pixels']
    area = args['area']
    if area == "north":
        max_set_len = 30
    elif area == "south":
        max_set_len = 25
    else:
        max_set_len = 40
    gal = args['gal_type']
    traindata, valdata, testdata = get_final_dataset(num_pixels=num_pixels, max_set_len=max_set_len, gal=gal, area=area)
    features = traindata.num_features


def print_session_stats(args):
    print()
    print('++++++++ Session Characteristics +++++++')
    print()
    print(f"Area: {area}")
    print(f"Gal Type: {gal}")
    print(f"Training Set: {traindata.num_pixels}")
    print(f"Validation Set: {valdata.num_pixels}")
    print(f"Test Samples: {testdata.num_pixels}")
    print(f"Number of features: {features}")
    print(f"Device: {device}")
    print(f"Number of Workers: {num_workers}")
    print()
    print('+++++++++++++++++++++++++++++++++++++++')


def define_model(galaxy, area):
    # ToDo: Provide cases for Dropout Galaxies
    # defines and returns the best models from HP Tuning
    if area == "north":
        if galaxy == 'lrg':
            n_layers_fe = 2
            out_features_fe = 150
            p1 = 0.3
            p2 = 0.25
            med_layer = 150
            n_layers_mlp = 4
            out_features_mlp = 100

        elif galaxy == 'elg':
            n_layers_fe = 2
            out_features_fe = 60
            p1 = 0.1
            p2 = 0.27
            med_layer = 380
            n_layers_mlp = 4
            out_features_mlp = 100

        elif galaxy == 'qso':
            n_layers_fe = 2
            out_features_fe = 175
            p1 = 0.075
            p2 = 0.25
            med_layer = 250
            n_layers_mlp = 4
            out_features_mlp = 175

        elif galaxy == 'glbg':
            n_layers_fe = 2
            out_features_fe = 70
            p1 = 0.2
            p2 = 0.25
            med_layer = 100
            n_layers_mlp = 4
            out_features_mlp = 200

        else:
            n_layers_fe = 2
            out_features_fe = 220
            p1 = 0.3
            p2 = 0.2
            med_layer = 150
            n_layers_mlp = 4
            out_features_mlp = 100

    elif area == "south":

        if galaxy == 'lrg':
            n_layers_fe = 2
            out_features_fe = 150
            p1 = 0.3
            p2 = 0.25
            med_layer = 150
            n_layers_mlp = 4
            out_features_mlp = 100

        elif galaxy == 'elg':
            n_layers_fe = 2
            out_features_fe = 250
            p1 = 0.25
            p2 = 0.3
            med_layer = 500
            n_layers_mlp = 4
            out_features_mlp = 175

        elif galaxy == 'qso':
            n_layers_fe = 2
            out_features_fe = 200
            p1 = 0.2
            p2 = 0.2
            med_layer = 185
            n_layers_mlp = 4
            out_features_mlp = 100

        elif galaxy == 'glbg':
            n_layers_fe = 2
            out_features_fe = 200
            p1 = 0.25
            p2 = 0.25
            med_layer = 400
            n_layers_mlp = 2
            out_features_mlp = 100

        else:
            n_layers_fe = 4
            out_features_fe = 150
            p1 = 0.25
            p2 = 0.05
            med_layer = 300
            n_layers_mlp = 2
            out_features_mlp = 175

    else:

        if galaxy == 'lrg':
            n_layers_fe = 2
            out_features_fe = 150
            p1 = 0.3
            p2 = 0.25
            med_layer = 150
            n_layers_mlp = 4
            out_features_mlp = 100

        elif galaxy == 'elg':
            n_layers_fe = 4
            out_features_fe = 150
            p1 = 0.2
            p2 = 0.25
            med_layer = 350
            n_layers_mlp = 4
            out_features_mlp = 160

        elif galaxy == 'qso':
            n_layers_fe = 2
            out_features_fe = 150
            p1 = 0.25
            p2 = 0.4
            med_layer = 20
            n_layers_mlp = 2
            out_features_mlp = 120

        elif galaxy == 'glbg':
            n_layers_fe = 4
            out_features_fe = 100
            p1 = 0.25
            p2 = 0.25
            med_layer = 130
            n_layers_mlp = 4
            out_features_mlp = 100

        else:
            n_layers_fe = 4
            out_features_fe = 180
            p1 = 0.25
            p2 = 0.35
            med_layer = 330
            n_layers_mlp = 2
            out_features_mlp = 200

    reduce = 'sum'
    fe_layers = []

    in_features = features

    for i in range(n_layers_fe):
        fe_layers.append(nn.Linear(in_features, out_features_fe))
        fe_layers.append(nn.ReLU())
        fe_layers.append(nn.Dropout(p1))
        in_features = out_features_fe

    # Getting Output Layer for FE that is then fed into Invariant Layer
    fe_layers.append(nn.Linear(out_features_fe, med_layer))
    fe_layers.append(nn.ReLU())

    mlp_layers = []

    in_features = 22

    for i in range(n_layers_mlp):
        mlp_layers.append(nn.Linear(in_features, out_features_mlp))
        mlp_layers.append(nn.ReLU())
        # if n_layers_mlp // 2 == i:
        mlp_layers.append(nn.Dropout(p2))
        in_features = out_features_mlp
    mlp_layers.append(nn.Linear(in_features, int(in_features / 2)))
    mlp_layers.append(nn.Linear(int(in_features / 2), 1))

    # mlp_layers.append(nn.ReLU())

    return VarMultiSetNet(feature_extractor=nn.Sequential(*fe_layers), mlp=nn.Sequential(*mlp_layers),
                          med_layer=med_layer, reduction=reduce)


def get_hparams(galaxy, area):
    # ToDo: Provide cases for dropout galaxies
    # defines and returns: lr, weight_decay, batch_size
    if area == "north":
        if galaxy == 'lrg': # --> done
            return 0.0001546764411255828, 0.03184751820920388, 32
        elif galaxy == 'elg':  # --> done
            return 0.0004599944376425736, 0.0738690886140696, 32

        elif galaxy == 'qso':  # --> done
            return 0.0002129782693067251, 0.01091575020984078, 32

        elif galaxy == 'glbg':  # --> done
            return 0.0010669093353509024, 0.12333702988885017, 128
        else: # --> done
            return 9.698438573821183e-05, 0.23961834074306818, 256

    elif area == "south":
        if galaxy == 'lrg': # --> done
            return 0.0001546764411255828, 0.03184751820920388, 32
        elif galaxy == 'elg':  # --> done
            return 0.00018641861416175164, 0.07947795784779363, 256
        elif galaxy == 'qso': # --> done
            return 0.0001883515972221876, 0.044348213953226155, 256
        elif galaxy == 'glbg':
            return 0.00030735807502983687, 0.2898483420649945, 32
        else:  # --> done
            return 0.00023951669569601602, 0.12288462842525326, 256
    else:
        if galaxy == 'lrg':  # --> done
            return 0.0001546764411255828, 0.03184751820920388, 32
        elif galaxy == 'elg': # --> done
            return 9.342861572715064e-05, 0.055109740552741496, 32
        elif galaxy == 'qso': # --> done
            return 0.005544184180774864, 0.008298229989336776, 32
        elif galaxy == 'glbg': # --> done
            return 0.00014553858586568378, 0.20049865047593457, 32
        else: # --> done
            return 6.646267123877951e-05, 0.003691296129378855, 256


if __name__ == "__main__":
    main()
