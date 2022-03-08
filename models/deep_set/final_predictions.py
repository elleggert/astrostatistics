import pandas as pd
import torch
import os
import numpy as np
import pickle
from datasets import MultiSetSequence
from torch.utils.data import DataLoader
from sklearn import metrics
import math
from util import get_mask

areas = ['north', 'south', 'des']
galaxies = ['lrg', 'elg', 'qso', 'glbg', 'rlbg']
device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
num_workers = 0 if device == 'cpu:0' else 8
NSIDE = 512

for area in areas:

    df_deep = None

    print(f'Area: {area} started loading.')
    with open(f'data/{area}/{area}_512_robust.pickle', 'rb') as f:
        trainset = pickle.load(f)
        f.close()
    with open(f'data/{area}/{area}_test_512_robust.pickle', 'rb') as f:
        testset = pickle.load(f)
        f.close()

    if area == "north":
        max_set_len = 30
    elif area == "south":
        max_set_len = 25
    else:
        max_set_len = 40
    df_test = pd.DataFrame.from_dict(testset, orient='index')
    df_train = pd.DataFrame.from_dict(trainset, orient='index')
    print(len(df_test), len(df_train))
    df_test = df_test.append(df_train)
    print(len(df_test))

    testdata = MultiSetSequence(dict=df_test.to_dict(orient='index'), num_pixels=len(df_test),
                                max_ccds=max_set_len, num_features=5, test=True)

    pixel_id = testdata.pixel_id

    print(f'Area: {area} finished loading.')
    print()

    for gal in galaxies:
        testdata.set_targets(gal_type=gal)

        best_val = -100
        for model in os.listdir(f"trained_models/{area}/{gal}"):
            try:
                int(model[:-3])
                continue

            except:
                val = float(model[:-3])
                if val > best_val:
                    best_val = val

        print()
        print()
        print(f' Area: {area}. Gal: {gal}. Best val: {best_val}.')
        print()

        if device == 'cpu:0':
            model = torch.load(f"trained_models/{area}/{gal}/{best_val}.pt",
                               map_location=torch.device('cpu'))
        else:
            model = torch.load(f"trained_models/{area}/{gal}/{best_val}.pt")

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

            print(len(y_pred))
            r2 = metrics.r2_score(y_gold, y_pred)
            rmse = math.sqrt(metrics.mean_squared_error(y_gold, y_pred))
            mae = metrics.mean_absolute_error(y_gold, y_pred)

            print()
            print(f" XXXXXX======== TRIAL {area} - {gal} ended")
            print()
            print("Test Set - R-squared: ", r2)
            print("Test Set - RMSE: ", rmse)
            print("Test Set - MAE: ", mae)

        ax = np.stack((pixel_id, y_pred), axis=1)

        if df_deep is None:
            df_deep = pd.DataFrame(ax, columns=['pixel_id', f'{gal}_deep'])
            df_deep.pixel_id = df_deep.pixel_id.astype(int)
        else:
            df_temp = pd.DataFrame(ax, columns=['pixel_id', f'{gal}_deep'])
            df_deep = df_deep.merge(df_temp, how='inner', on='pixel_id')

    df_deep = df_deep.dropna()
    df_deep.to_csv(f'results/{area}_ds_predictions.csv', index=False)
    print(f' Pixels in Area: {area}: {len(df_deep)}. ')
