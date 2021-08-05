import argparse

import numpy as np
import optuna
import torch
from optuna.trial import TrialState
from sklearn import metrics
from torch import nn, optim
from torch.utils.data import DataLoader


from models import BaseNet
from util import get_dataset, get_full_dataset

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

    print(study.trials)
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


    model = torch.load(f'trained_models/{gal}/{trial.value}.pt', map_location=torch.device('cpu')) # Delete later

    testloader = torch.utils.data.DataLoader(valdata, batch_size=128, shuffle=False)

    mse, r2 = 0, 0

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

        r2 = metrics.r2_score(y_gold, y_pred)
        mse = metrics.mean_squared_error(y_gold, y_pred)

        print("Test Set - R-squared: ", r2)
        print("Test Set - MSE: ", mse)




def parse_command_line_args(args):
    global gal, num_pixels, path_to_data, max_set_len, traindata, valdata, testdata
    num_pixels = args['num_pixels']
    gal = args['gal_type']
    traindata, valdata, testdata = get_full_dataset(gal=gal)



def print_session_stats(args):
    print('++++++++ Session Characteristics +++++++')
    print()
    print(f"Gal Type: {gal}")
    print(f"Training Samples: {len(traindata)}")
    print(f"Validation Samples: {len(valdata)}")
    print(f"Test Samples: {len(testdata)}")
    print(f"Device: {device}")
    print(f"Number of Workers: {num_workers}")
    print(f"Number of Trials: {args['trials']}")
    print()
    print('+++++++++++++++++++++++++++++++++++++++')


def define_model(trial):
    n_layers_mlp = trial.suggest_int("n_layers_mlp", low=2, high=8, step=2)
    mlp_layers = []

    in_features = 21 # --> make a function argument later

    for i in range(n_layers_mlp):
        out_features = trial.suggest_int("mlp_n_units_l{}".format(i), 8, 400)
        mlp_layers.append(nn.Linear(in_features, out_features))
        mlp_layers.append(nn.ReLU())
        #if n_layers_mlp // 2 == i:
        p = trial.suggest_float("mlp_dropout_l{}".format(i), 0.0, 0.5)
        mlp_layers.append(nn.Dropout(p))

        in_features = out_features

    mlp_layers.append(nn.Linear(in_features, 1))
    mlp_layers.append(nn.ReLU())

    return BaseNet(mlp=nn.Sequential(*mlp_layers))


def objective(trial):
    model = define_model(trial).to(device)
    print(
        f"Trial Id: {trial.number} | Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)} | Timestamp: {trial.datetime_start}")
    print()



    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    criterion_name = trial.suggest_categorical("criterion", ["MSELoss", "L1Loss"])
    criterion = getattr(nn, criterion_name)()

    batch_size = trial.suggest_categorical("batch_size", [16, 32, 128, 256, 1028])

    drop_last = True if (len(valdata.input) > batch_size) else False
    no_epochs = trial.suggest_int("no_epochs", 30, 300)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers, drop_last=drop_last)

    valloader = torch.utils.data.DataLoader(valdata, batch_size=batch_size, shuffle=False, drop_last=drop_last)

    mse, r2 = 0, 0


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

                #Split dataloader
                inputs = inputs.to(device)
                #Forward pass through the trained network
                outputs = model(inputs)

                #Get predictions and append to label array + count number of correct and total
                y_pred = np.append(y_pred, outputs.cpu().detach().numpy())
                y_gold = np.append(y_gold, labels.cpu().detach().numpy())


        try:
            r2 = metrics.r2_score(y_gold, y_pred)
            mse = metrics.mean_squared_error(y_gold, y_pred)
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

        torch.save(model, f"trained_models/{gal}/{trial.number}.pt")
    return r2


if __name__ == "__main__":
    main()

