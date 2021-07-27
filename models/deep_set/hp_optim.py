import optuna


def define_model(trial):

    n_layers_fe = trial.suggest_int("n_layers_fe", low=2, high=8, step=2)
    n_layers_mlp = trial.suggest_int("n_layers_mlp", low=2, high=8, step=2)
