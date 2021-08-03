

import argparse

import numpy as np
import optuna
import pytorch_lightning as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback
from optuna.trial import TrialState
from pytorch_lightning.callbacks import EarlyStopping
from sklearn import metrics
from torch import nn, optim
from torch.utils.data import DataLoader


from models import VarMultiSetNet
from lightning import LitVarDeepSet, DeepDataModule
from util import get_dataset, get_mask

device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
num_workers = 8 if device == 'cpu:0' else 8


def main():
    parser = argparse.ArgumentParser(description='MultiSetSequence DeepSet-Network - HyperParameter Tuning',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--path_to_data', default='../../bricks_data/multiset.pickle', metavar='', type=str,
                        help='path to the data directory')
    parser.add_argument('-n', '--num_pixels', default=1500, metavar='', type=int, help='number of training examples')
    parser.add_argument('-c', '--max_ccds', default=30, metavar='', type=int,
                        help='Maximum set lengths for individual CCDs')
    parser.add_argument('-g', '--gal_type', default='lrg', metavar='', type=str, help='Galaxy Type to optimise model for')
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

    fig1 = optuna.visualization.plot_optimization_history(study, target_name=f'R-squared for {gal}-optimisation ')
    fig1.write_image(f"logs_figs/hp_search_{gal}.png")


def parse_command_line_args(args):
    global gal, num_pixels, path_to_data, max_set_len, datamodule
    num_pixels = args['num_pixels']
    path_to_data = args['path_to_data']
    max_set_len = args['max_ccds']
    gal = args['gal_type']
    datamodule = DeepDataModule(num_pixels=num_pixels, max_set_len=max_set_len,gal=gal,path_to_data=path_to_data)



def print_session_stats(args):
    print('++++++++ Session Characteristics +++++++')
    print()
    print(f"Gal Type: {gal}")
    print(f"Training Set: {datamodule.traindata.num_pixels}")
    print(f"Validation Set: {datamodule.valdata.num_pixels}")
    print(f"Maximum Set Lengths: {max_set_len}")
    print(f"Device: {device}")
    print(f"Number of Workers: {num_workers}")
    print(f"Number of Trials: {args['trials']}")
    print()
    print('+++++++++++++++++++++++++++++++++++++++')


def define_model(trial):
    n_layers_fe = trial.suggest_int("n_layers_fe", low=2, high=8, step=2)

    fe_layers = []

    in_features = 15  # --> make a function argument later

    for i in range(n_layers_fe):
        out_features = trial.suggest_int("fe_n_units_l{}".format(i), 32, 256) # ToDo Larger --> experiment
        fe_layers.append(nn.Linear(in_features, out_features))
        fe_layers.append(nn.ReLU())
        if n_layers_fe // 2 == i:
            p = trial.suggest_float("fe_dropout_l{}".format(i), 0.0, 0.5) # Experiment with more dropout
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
        out_features = trial.suggest_int("mlp_n_units_l{}".format(i), 32, 256)
        mlp_layers.append(nn.Linear(in_features, out_features))
        mlp_layers.append(nn.ReLU())
        if n_layers_mlp // 2 == i:
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


    batch_size = trial.suggest_categorical("batch_size", [16,32,128])

    no_epochs = 300 # --> Get rid of it , Early stopping ToDo

    datamodule.batch_size = batch_size

    trainer = pl.Trainer(logger=False,
        checkpoint_callback=False,
        max_epochs=no_epochs,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="Val_loss"), EarlyStopping(monitor='Val_loss', patience=10)])

    trainer.fit(model, datamodule=datamodule)

    #torch.save(model, "trained_models/{}.pt".format(trial.number))

    return trainer.callback_metrics["Val_loss"].item()



if __name__ == "__main__":
    main()

