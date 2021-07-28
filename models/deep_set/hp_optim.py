import random

import optuna
import torch
from optuna.trial import TrialState
from torch import nn

from models import VarMultiSetNet

device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'


def define_model(trial):
    n_layers_fe = trial.suggest_int("n_layers_fe", low=2, high=8, step=2)

    fe_layers = []

    in_features = 15  # --> make a function argument later

    for i in range(n_layers_fe):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        fe_layers.append(nn.Linear(in_features, out_features))
        fe_layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.0, 0.5)
        fe_layers.append(nn.Dropout(p))

        in_features = out_features

    # Getting Output Layer for FE that is then fed into Invariant Layer
    med_layer = trial.suggest_int("n_units_l{}".format(1), 16, 64)
    fe_layers.append(nn.Linear(in_features, med_layer))
    fe_layers.append(nn.ReLU())

    n_layers_mlp = trial.suggest_int("n_layers_mlp", low=2, high=8, step=2)
    mlp_layers = []

    in_features = 66

    for i in range(n_layers_mlp):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        mlp_layers.append(nn.Linear(in_features, out_features))
        mlp_layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.0, 0.5)
        mlp_layers.append(nn.Dropout(p))

        in_features = out_features

    mlp_layers.append(nn.Linear(in_features, 1))
    mlp_layers.append(nn.ReLU())

    reduction = trial.suggest_categorical("reduction", ["sum", "mean", "min", "max"])
    return VarMultiSetNet(feature_extractor=nn.Sequential(*fe_layers), mlp=nn.Sequential(*mlp_layers),
                          med_layer=med_layer, reduction=reduction)


def objective(trial):
    model = define_model(trial).to(device)

    return random.uniform(0, 1)


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="DeepSet")

    study.optimize(objective, n_trials=5, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
