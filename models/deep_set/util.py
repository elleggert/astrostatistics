import pickle

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import MultiSetSequence


# General Utility Functions used in Model Training and For HyperParameter Optimisation --> factored out to have clearer HP optim file

def get_mask(sizes, max_size):
    return (torch.arange(max_size).reshape(1, -1).to(sizes.device) < sizes.unsqueeze(2))

def get_full_dataset(area, num_pixels, max_set_len, gal):
    print(f"Starting Loading {area}")
    with open(f'data/{area}/{area}_512_robust.pickle', 'rb') as f:
        trainset = pickle.load(f)
        f.close()
    with open(f'data/{area}/{area}_test_512_robust.pickle', 'rb') as f:
        testset = pickle.load(f)
        f.close()

    print(f"Finished Loading {area}")

    df_train = pd.DataFrame.from_dict(trainset, orient='index')
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=666, shuffle=True)

    df_test = pd.DataFrame.from_dict(testset, orient='index')

    if num_pixels is None:
        num = len(df_train) + len(df_val)
    else:
        num = num_pixels
        if num > (len(df_train) + len(df_val)):
            num = len(df_train) + len(df_val)

    traindata = MultiSetSequence(dict=df_train.to_dict(orient='index'), num_pixels=round(num),
                                 max_ccds=max_set_len, num_features=5)
    traindata.set_targets(gal_type=gal)

    valdata = MultiSetSequence(dict=df_val.to_dict(orient='index'), num_pixels=round(num * 0.2),
                               max_ccds=max_set_len, num_features=5)
    valdata.set_targets(gal_type=gal)

    testdata = MultiSetSequence(dict=df_test.to_dict(orient='index'), num_pixels=len(df_test),
                                max_ccds=max_set_len, num_features=5)
    testdata.set_targets(gal_type=gal)

    print(f"Finished {area} setup")

    return traindata, valdata, testdata


def get_final_dataset(area, num_pixels, max_set_len, gal):
    print(f"Starting Loading {area}")
    with open(f'data/{area}/{area}_512_robust.pickle', 'rb') as f:
        trainset = pickle.load(f)
        f.close()
    with open(f'data/{area}/{area}_test_512_robust.pickle', 'rb') as f:
        testset = pickle.load(f)
        f.close()

    print(f"Finished Loading {area}")

    df_train = pd.DataFrame.from_dict(trainset, orient='index')
    df_test = pd.DataFrame.from_dict(testset, orient='index')

    if num_pixels is None:
        num = len(df_train)
    else:
        num = num_pixels

    traindata = MultiSetSequence(dict=df_train.to_dict(orient='index'), num_pixels=round(num),
                                 max_ccds=max_set_len, num_features=5)
    traindata.set_targets(gal_type=gal)

    testdata = MultiSetSequence(dict=df_test.to_dict(orient='index'), num_pixels=len(df_test),
                                max_ccds=max_set_len, num_features=5)
    testdata.set_targets(gal_type=gal)

    print(f"Finished {area} setup")

    return traindata, testdata, testdata