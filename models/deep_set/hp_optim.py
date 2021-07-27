import optuna
from torch import nn

from models import MultiSetNet, VarMultiSetNet

feature_extractor = nn.Sequential(
    nn.Linear(15, 32),
    nn.ReLU(inplace=False),
    nn.Linear(32, 128),
    nn.ReLU(inplace=False),
    nn.Dropout(p=0.7, inplace=False),
    nn.Linear(128, 64),
    nn.ReLU(inplace=False),
    nn.Linear(64, 16),
    nn.ReLU(inplace=False)
)
mlp = nn.Sequential(
    nn.Linear(64 + 2, 128),
    nn.ReLU(inplace=False),
    nn.Linear(128, 64),
    nn.ReLU(inplace=False),
    nn.Dropout(p=0.7, inplace=False),
    nn.Linear(64, 32),
    nn.ReLU(inplace=False),
    nn.Linear(32, 1),
    nn.ReLU(inplace=False)
)

model = MultiSetNet()
model2 = VarMultiSetNet(feature_extractor=feature_extractor, mlp=mlp)

print(model)
print(model2)

def define_model(trial):

    n_layers_fe = trial.suggest_int("n_layers_fe", low=2, high=8, step=2)

    fe_layers = []

    in_features = 15 # --> make a function argument later

    med_features = trial.suggest_int("n_units_l{}".format(1), 16, 64)

    for i in range(n_layers_fe):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        fe_layers.append(nn.Linear(in_features, out_features))
        fe_layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.0, 0.5)
        fe_layers.append(nn.Dropout(p))

        in_features = out_features

    fe_layers.append(nn.Linear(in_features, med_features))
    fe_layers.append(nn.ReLU())






    n_layers_mlp = trial.suggest_int("n_layers_mlp", low=2, high=8, step=2)
    mlp_layers = []
