"""Hyperparameter Search Function without Hp_lightning, with optuna. Simply trains models until completion,
notes optimisation history as log and visually in /logs_figs"""

import argparse
import math
import os

import numpy as np
import optuna
import torch
from optuna.trial import TrialState
from sklearn import metrics
from torch import nn, optim
from torch.utils.data import DataLoader

from models import VarMultiSetNet
from util import get_mask, get_full_dataset

device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
num_workers = 0 if device == 'cpu:0' else 8


def main():
    parser = argparse.ArgumentParser(description='MultiSetSequence DeepSet-Network - HyperParameter Tuning',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    """parser.add_argument('-d', '--path_to_data', default='data/multiset.pickle', metavar='', type=str,
                        help='path to the data directory')"""
    parser.add_argument('-n', '--num_pixels', default=None, metavar='', type=int, help='number of training examples')
    """parser.add_argument('-c', '--max_ccds', default=30, metavar='', type=int,
                        help='Maximum set lengths for individual CCDs')"""
    parser.add_argument('-a', '--area', default='north', metavar='', type=str,
                        help='The area of the sky that should be trained on')
    parser.add_argument('-g', '--gal_type', default='lrg', metavar='', type=str,
                        help='Galaxy Type to optimise model for')
    parser.add_argument('-t', '--trials', default=200, metavar='', type=int, help='number of trials to tune HP for')

    args = vars(parser.parse_args())

    parse_command_line_args(args)

    print_session_stats(args)

    study = optuna.create_study(directions=["minimize"], study_name="DeepSet")

    study.optimize(objective, n_trials=args['trials'], timeout=None)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print()
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print()
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")

    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    fig1 = optuna.visualization.plot_optimization_history(study,
                                                          target_name=f'RMSE-squared for {gal}-{area}-optimisation ')
    fig2 = optuna.visualization.plot_param_importances(study)
    #fig1.write_image(f"logs_figs/{area}/hp_search_{gal}.png")
    #fig2.write_image(f"logs_figs/{area}/hp_params_{gal}.png")

    if device == 'cpu:0':
        model = torch.load(f"trained_models/{area}/{gal}/{trial.number}.pt",
                           map_location=torch.device('cpu'))
    else:
        model = torch.load(f"trained_models/{area}/{gal}/{trial.number}.pt")

    delete_models()

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


def delete_models():
    for model in os.listdir(f"trained_models/{area}/{gal}"):
        os.remove(f"trained_models/{area}/{gal}/{model}")


def parse_command_line_args(args):
    global gal, area, num_pixels, max_set_len, traindata, valdata, testdata, features
    num_pixels = args['num_pixels']
    # path_to_data = args['path_to_data']
    area = args['area']
    if area == "north":
        max_set_len = 30
    elif area == "south":
        max_set_len = 25
    else:
        max_set_len = 50
    gal = args['gal_type']
    traindata, valdata, testdata = get_full_dataset(num_pixels=num_pixels, max_set_len=max_set_len, gal=gal, area=area)
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
    print(f"Number of Trials: {args['trials']}")
    print()
    print('+++++++++++++++++++++++++++++++++++++++')


def define_model(trial):
    n_layers_fe = trial.suggest_int("n_layers_fe", low=2, high=4, step=2)

    fe_layers = []

    in_features = features

    for i in range(n_layers_fe):
        out_features = trial.suggest_int("fe_n_units_l{}".format(i), 8, 100)  # 256
        fe_layers.append(nn.Linear(in_features, out_features))
        fe_layers.append(nn.ReLU())
        # if n_layers_fe // 2 == i:
        p = trial.suggest_float("fe_dropout_l{}".format(i), 0.0, 0.5)  # Experiment with more dropout
        fe_layers.append(nn.Dropout(p))

        in_features = out_features

    # Getting Output Layer for FE that is then fed into Invariant Layer
    med_layer = trial.suggest_int("n_units_l{}".format('(Invariant)'), 1, 100)  # 512
    fe_layers.append(nn.Linear(in_features, med_layer))
    fe_layers.append(nn.ReLU())

    n_layers_mlp = trial.suggest_int("n_layers_mlp", low=2, high=4, step=2)
    mlp_layers = []

    in_features = 22

    for i in range(n_layers_mlp):
        out_features = trial.suggest_int("mlp_n_units_l{}".format(i), 8, 100)  # 256
        mlp_layers.append(nn.Linear(in_features, out_features))
        mlp_layers.append(nn.ReLU())
        # if n_layers_mlp // 2 == i:
        p = trial.suggest_float("mlp_dropout_l{}".format(i), 0.0, 0.5)
        mlp_layers.append(nn.Dropout(p))

        in_features = out_features

    mlp_layers.append(nn.Linear(in_features, 1))
    mlp_layers.append(nn.ReLU())

    reduce = trial.suggest_categorical("reduction", ["sum", "mean", "max"])
    return VarMultiSetNet(feature_extractor=nn.Sequential(*fe_layers), mlp=nn.Sequential(*mlp_layers),
                          med_layer=med_layer, reduction=reduce)


def objective(trial):
    model = define_model(trial).to(device)
    print()
    print(
        f"Trial Id: {trial.number} | Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)} "
        f"| Timestamp: {trial.datetime_start}")
    print()
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)
    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_name = trial.suggest_categorical("criterion", ["MSELoss", "PoissonNLLLoss"])
    criterion = getattr(nn, criterion_name)()
    # criterion = nn.MSELoss()

    batch_size = trial.suggest_categorical("batch_size", [32, 128, 256])

    drop_last = True if (len(valdata.input) > batch_size) else False
    no_epochs = 100

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers, drop_last=drop_last)

    valloader = torch.utils.data.DataLoader(valdata, batch_size=batch_size, shuffle=False, drop_last=False)

    rmse, r2 = 0, 0

    for epoch in range(no_epochs):

        model.train()

        train_loop(criterion, model, optimiser, trainloader)

        model.eval()

        y_gold, y_pred = val_loop(model, valloader, y_gold, y_pred)

        try:
            r2 = metrics.r2_score(y_gold, y_pred)
            rmse = math.sqrt(metrics.mean_squared_error(y_gold, y_pred))
            if epoch % 5 == 0:
                print("epoch", epoch, r2, rmse)

        except:
            print("++++++++++++++++++++")
            print("        NaN         ")
            print("++++++++++++++++++++")
            trial.report(-100, epoch)
            raise optuna.exceptions.TrialPruned()

        trial.report(r2, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        torch.save(model, f"trained_models/{area}/{gal}/{trial.number}.pt")

    return rmse


def val_loop(model, valloader):
    y_pred, y_gold = np.array([]), np.array([])
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


if __name__ == "__main__":
    main()
