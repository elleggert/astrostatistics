import argparse

import numpy as np
import optuna
import torch
from optuna.trial import TrialState
from sklearn import metrics
from torch import nn, optim
from torch.utils.data import DataLoader


from models import BaseNet
from util import get_dataset

device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
num_workers = 0 if device == 'cpu:0' else 8


def main():
    parser = argparse.ArgumentParser(description='MBase-Network using Average Systematics - HyperParameter Tuning',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--num_pixels', default=None, metavar='', type=int, help='number of training examples')
    parser.add_argument('-g', '--gal_type', default='lrg', metavar='', type=str, help='Galaxy Type to optimise model for')
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
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")

    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    fig1 = optuna.visualization.plot_optimization_history(study, target_name=f'R-squared for {gal}-optimisation ')
    fig1.write_image(f"logs_figs/hp_search_{gal}.png")

def parse_command_line_args(args):
    global gal, num_pixels, path_to_data, max_set_len, traindata, valdata
    num_pixels = args['num_pixels']
    path_to_data = args['path_to_data']
    max_set_len = args['max_ccds']
    gal = args['gal_type']
    traindata, valdata = get_dataset(num_pixels=num_pixels, max_set_len=max_set_len, gal=gal, path_to_data=path_to_data)



def print_session_stats(args):
    print('++++++++ Session Characteristics +++++++')
    print()
    print(f"Gal Type: {gal}")
    print(f"Training Samples: {traindata.num_pixels}")
    print(f"Validation Samples: {valdata.num_pixels}")
    print(f"Maximum Set Lengths: {max_set_len}")
    print(f"Device: {device}")
    print(f"Number of Workers: {num_workers}")
    print(f"Number of Trials: {args['trials']}")
    print()
    print('+++++++++++++++++++++++++++++++++++++++')
