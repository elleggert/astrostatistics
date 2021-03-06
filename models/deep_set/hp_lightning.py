"""Utilising HP_Lightning to integrate early stopping into the Optuna Pruner"""

import argparse
import math
import os

import numpy as np
import optuna
import pytorch_lightning as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback
from optuna.trial import TrialState
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import metrics
from torch import nn, optim
from torch.utils.data import DataLoader

from models import VarMultiSetNet
from lightning import LitVarDeepSet, DeepDataModule
from util import get_dataset, get_mask

device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
num_workers = 0 if device == 'cpu:0' else 8


def main():
    parser = argparse.ArgumentParser(description='MultiSetSequence DeepSet-Network - HyperParameter Tuning',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--path_to_data', default='data/multiset.pickle', metavar='', type=str,
                        help='path to the data directory')
    parser.add_argument('-n', '--num_pixels', default=1500, metavar='', type=int, help='number of training examples')
    parser.add_argument('-c', '--max_ccds', default=30, metavar='', type=int,
                        help='Maximum set lengths for individual CCDs')
    parser.add_argument('-a', '--area', default='des', metavar='', type=str,
                        help='The area of the sky that should be trained on')
    parser.add_argument('-g', '--gal_type', default='lrg', metavar='', type=str,
                        help='Galaxy Type to optimise model for')
    parser.add_argument('-t', '--trials', default=200, metavar='', type=int, help='number of trials to tune HP for')

    args = vars(parser.parse_args())

    parse_command_line_args(args)

    print_session_stats(args)

    study = optuna.create_study(directions=["maximize"], study_name="DeepSet")

    study.optimize(objective, n_trials=args['trials'], timeout=None)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print()
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print()
    print("Best trial by Validation Set R-squared:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")

    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    fig1 = optuna.visualization.plot_optimization_history(study, target_name=f'R-squared for {gal}-optimisation ')
    fig1.write_image(f"logs_figs/{area}/hp_search_{gal}.png")

    if device == 'cpu:0':
        model = torch.load(f"trained_models/{area}/{gal}/{trial.number}.pt",
                           map_location=torch.device('cpu'))
    else:
        model = torch.load(f"trained_models/{area}/{gal}/{trial.number}.pt")

    delete_models()

    testloader = torch.utils.data.DataLoader(datamodule.testdata, batch_size=128, shuffle=False)

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

        r2 = metrics.r2_score(y_gold, y_pred)
        rmse = math.sqrt(metrics.mean_squared_error(y_gold, y_pred))

        print()
        print("Test Set - R-squared: ", r2)
        print("Test Set - RMSE: ", rmse)

    torch.save(model, f"trained_models/{area}/{gal}/{r2}.pt")


def delete_models():
    for model in os.listdir(f"trained_models/{area}/{gal}"):
        os.remove(f"trained_models/{area}/{gal}/{model}")


def parse_command_line_args(args):
    global gal, area, num_pixels, path_to_data, datamodule, features
    num_pixels = args['num_pixels']
    area = args['area']
    path_to_data = args['path_to_data']
    gal = args['gal_type']
    datamodule = DeepDataModule(area=area, num_pixels=num_pixels, gal=gal, path_to_data=path_to_data)
    features = datamodule.num_features


def print_session_stats(args):
    print('++++++++ Session Characteristics +++++++')
    print()
    print(f"Area: {area}")
    print(f"Gal Type: {gal}")
    print(f"Training Set: {datamodule.traindata.num_pixels}")
    print(f"Validation Set: {datamodule.valdata.num_pixels}")
    print(f"Test Samples: {datamodule.testdata.num_pixels}")
    print(f"Number of features: {features}")
    print(f"Device: {device}")
    print(f"Number of Workers: {num_workers}")
    print(f"Number of Trials: {args['trials']}")
    print()
    print('+++++++++++++++++++++++++++++++++++++++')


def define_model(trial):
    n_layers_fe = trial.suggest_int("n_layers_fe", low=2, high=8, step=2)

    fe_layers = []

    in_features = features

    for i in range(n_layers_fe):
        out_features = trial.suggest_int("fe_n_units_l{}".format(i), 8, 16)
        fe_layers.append(nn.Linear(in_features, out_features))
        fe_layers.append(nn.ReLU())
        # if n_layers_fe // 2 == i:
        p = trial.suggest_float("fe_dropout_l{}".format(i), 0.0, 0.5)  # Experiment with more dropout
        fe_layers.append(nn.Dropout(p))

        in_features = out_features

    # Getting Output Layer for FE that is then fed into Invariant Layer
    med_layer = trial.suggest_int("n_units_l{}".format('(Invariant)'), 16, 64)
    fe_layers.append(nn.Linear(in_features, med_layer))
    fe_layers.append(nn.ReLU())

    n_layers_mlp = trial.suggest_int("n_layers_mlp", low=2, high=8, step=2)
    mlp_layers = []

    in_features = 66

    for i in range(n_layers_mlp):
        out_features = trial.suggest_int("mlp_n_units_l{}".format(i), 8, 16)
        mlp_layers.append(nn.Linear(in_features, out_features))
        mlp_layers.append(nn.ReLU())
        p = trial.suggest_float("mlp_dropout_l{}".format(i), 0.0, 0.5)
        mlp_layers.append(nn.Dropout(p))

        in_features = out_features

    mlp_layers.append(nn.Linear(in_features, 1))
    mlp_layers.append(nn.ReLU())

    reduce = trial.suggest_categorical("reduction", ["sum", "mean"])
    return VarMultiSetNet(feature_extractor=nn.Sequential(*fe_layers), mlp=nn.Sequential(*mlp_layers),
                          med_layer=med_layer, reduction=reduce)


def objective(trial):
    net = define_model(trial)

    print(f"Trial Id: {trial.number} | Timestamp: {trial.datetime_start}")
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    criterion_name = trial.suggest_categorical("criterion", ["MSELoss", "L1Loss"])
    criterion = getattr(nn, criterion_name)()

    model = LitVarDeepSet(model=net, lr=lr, criterion=criterion)

    print()

    batch_size = 4  # trial.suggest_categorical("batch_size", [16,32,128,256])

    no_epochs = 200  # --> Get rid of it , Early stopping ToDo

    datamodule.batch_size = batch_size

    checkpoint_callback = ModelCheckpoint(monitor='Val_loss',
                                          dirpath=f'trained_models/{area}/{gal}/',
                                          filename='{Val_loss:.4f}',
                                          save_top_k=1,
                                          mode='min',
                                          every_n_epochs=1)

    # Have left pretty high patience since all the best models are stored
    trainer = pl.Trainer(logger=False,
                         checkpoint_callback=True,
                         max_epochs=no_epochs,
                         gpus=1 if torch.cuda.is_available() else None,
                         callbacks=[checkpoint_callback, PyTorchLightningPruningCallback(trial, monitor="Val_loss"),
                                    EarlyStopping(monitor='Val_loss', patience=15)])

    trainer.fit(model, datamodule=datamodule)

    # automatically auto-loads the best weights
    trainer.test(datamodule=datamodule)

    # torch.save(model, "rer/{}.pt".format(trial.number))

    return trainer.callback_metrics["Val_loss"].item()


if __name__ == "__main__":
    main()
