import argparse
import random

import numpy as np
import optuna
import torch
from optuna.trial import TrialState
from sklearn import metrics
from torch import nn, optim
from torch.utils.data import DataLoader


from models import VarMultiSetNet
from util import get_dataset, get_mask

gal = 'qso'
#gal = 'elg'
#gal = 'qso'


device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
num_workers = 0 if device == 'cpu:0' else 8
num_pixels = 25
max_set_len = 30
path_to_data='../../bricks_data/multiset.pickle'
traindata, valdata = get_dataset(num_pixels=num_pixels, max_set_len=max_set_len, gal=gal, path_to_data=path_to_data)



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

    study = optuna.create_study(directions=["maximize"], study_name="DeepSet")

    study.optimize(objective, n_trials=args['trials'], timeout=None)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")

    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    fig1 = optuna.visualization.plot_optimization_history(study)
    # fig2 = optuna.visualization.plot_intermediate_values(study)
    fig1.write_image("logs_figs/hp_search.png")


def parse_command_line_args(args):
    hp = globals()

    # Fix to use variable number of pixels in the future
    hp['num_pixels'] = args['num_pixels']
    hp['path_to_data'] = args['path_to_data']
    hp['max_set_len'] = args['max_ccds']
    hp['gal'] = args['gal_type']


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


def define_model(trial):
    n_layers_fe = trial.suggest_int("n_layers_fe", low=2, high=8, step=2)

    fe_layers = []

    in_features = 15  # --> make a function argument later

    for i in range(n_layers_fe):
        out_features = trial.suggest_int("fe_n_units_l{}".format(i), 8, 128)
        fe_layers.append(nn.Linear(in_features, out_features))
        fe_layers.append(nn.ReLU())
        p = trial.suggest_float("fe_dropout_l{}".format(i), 0.0, 0.3)
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
        out_features = trial.suggest_int("mlp_n_units_l{}".format(i), 8, 128)
        mlp_layers.append(nn.Linear(in_features, out_features))
        mlp_layers.append(nn.ReLU())
        p = trial.suggest_float("mlp_dropout_l{}".format(i), 0.0, 0.3)
        mlp_layers.append(nn.Dropout(p))

        in_features = out_features

    mlp_layers.append(nn.Linear(in_features, 1))
    mlp_layers.append(nn.ReLU())

    reduce = trial.suggest_categorical("reduction", ["sum", "mean"])
    return VarMultiSetNet(feature_extractor=nn.Sequential(*fe_layers), mlp=nn.Sequential(*mlp_layers),
                          med_layer=med_layer, reduction=reduce)


def objective(trial):
    model = define_model(trial).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}.")
    #print(model)


    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    criterion_name = trial.suggest_categorical("criterion", ["MSELoss", "L1Loss"])
    criterion = getattr(nn, criterion_name)()

    batch_size = trial.suggest_categorical("batch_size", [8,16,32,128,256])

    drop_last = True if (len(valdata.input) > batch_size) else False
    no_epochs = trial.suggest_int("no_epochs", 10, 300, log=True)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers, drop_last=drop_last)

    valloader = torch.utils.data.DataLoader(valdata, batch_size=batch_size, shuffle=False, drop_last=drop_last)

    mse, r2 = 0, 0

    for epoch in range(no_epochs):

        model.train()

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

        model.eval()
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

        try:
            r2 =  metrics.r2_score(y_gold, y_pred)
            mse = metrics.mean_squared_error(y_gold, y_pred)
        except:
            print("++++++++++++++++++++")
            print()
            print("NaN in epoch", epoch)
            print()
            print(model)
            print()
            print("++++++++++++++++++++")
            trial.report(-100, epoch)
            raise optuna.exceptions.TrialPruned()

        trial.report(r2, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return r2




if __name__ == "__main__":
    main()

