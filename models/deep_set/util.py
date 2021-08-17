import pickle

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import MultiSetSequence

# General Utility Functions used in Model Training and For HyperParameter Optimisation --> factored out to have clearer HP optim file

def get_mask(sizes, max_size):
    return (torch.arange(max_size).reshape(1, -1).to(sizes.device) < sizes.unsqueeze(2))


def get_dataset(num_pixels, max_set_len,gal, path_to_data='data/multiset.pickle'):
    with open(path_to_data, 'rb') as f:
        mini_multiset = pickle.load(f)
        f.close()
    df = pd.DataFrame.from_dict(mini_multiset, orient='index')
    zscore = lambda x: abs((x - x.median()) / x.std())
    df['Z_LRG'] = df[2].transform(zscore)
    df['Z_ELG'] = df[3].transform(zscore)
    df['Z_QSO'] = df[4].transform(zscore)
    df = df[(df['Z_LRG'] < 3)]
    df = df[(df['Z_ELG'] < 3)]
    df = df[(df['Z_QSO'] < 3)]
    num = num_pixels
    if num > len(df):
        num = len(df)
    df.drop(columns=['Z_ELG', 'Z_LRG','Z_QSO'], inplace=True)
    train_df, test_df = train_test_split(df, test_size=0.33, random_state=44, shuffle=True)
    traindata = MultiSetSequence(dict=train_df.to_dict(orient='index'), num_pixels=round(num * 0.67),
                                 max_ccds=max_set_len)
    traindata.set_targets(gal_type=gal)

    testdata = MultiSetSequence(dict=test_df.to_dict(orient='index'), num_pixels=round(num * 0.33),
                                max_ccds=max_set_len)
    testdata.set_targets(gal_type=gal, scaler=traindata.scaler)


    return traindata, testdata


def get_dataset(num_pixels, max_set_len,gal, path_to_data='data/multiset.pickle'):
    with open(path_to_data, 'rb') as f:
        mini_multiset = pickle.load(f)
        f.close()
    df = pd.DataFrame.from_dict(mini_multiset, orient='index')
    zscore = lambda x: abs((x - x.median()) / x.std())
    df['Z_LRG'] = df[2].transform(zscore)
    df['Z_ELG'] = df[3].transform(zscore)
    df['Z_QSO'] = df[4].transform(zscore)
    df = df[(df['Z_LRG'] < 3)]
    df = df[(df['Z_ELG'] < 3)]
    df = df[(df['Z_QSO'] < 3)]
    num = num_pixels
    if num > len(df):
        num = len(df)
    df.drop(columns=['Z_ELG', 'Z_LRG','Z_QSO'], inplace=True)
    train_df, test_df = train_test_split(df, test_size=0.33, random_state=44, shuffle=True)
    traindata = MultiSetSequence(dict=train_df.to_dict(orient='index'), num_pixels=round(num * 0.67),
                                 max_ccds=max_set_len)
    traindata.set_targets(gal_type=gal)

    testdata = MultiSetSequence(dict=test_df.to_dict(orient='index'), num_pixels=round(num * 0.33),
                                max_ccds=max_set_len)
    testdata.set_targets(gal_type=gal, scaler=traindata.scaler)


    return traindata, testdata


def get_full_dataset(area, num_pixels, max_set_len, gal):
    print(f"Starting Loading {area}")
    with open(f'data/{area}/{area}.pickle', 'rb') as f:
        trainset = pickle.load(f)
        f.close()
    with open(f'data/{area}/{area}_test.pickle', 'rb') as f:
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
    traindata = MultiSetSequence(dict=df_train.to_dict(orient='index'), num_pixels=round(num * 0.8),
                                 max_ccds=max_set_len,num_features=6)
    traindata.set_targets(gal_type=gal)

    valdata = MultiSetSequence(dict=df_val.to_dict(orient='index'), num_pixels=round(num * 0.2),
                                 max_ccds=max_set_len, num_features=6)
    valdata.set_targets(gal_type=gal)

    testdata = MultiSetSequence(dict=df_test.to_dict(orient='index'), num_pixels=len(df_test),
                               max_ccds=max_set_len, num_features=6)
    testdata.set_targets(gal_type=gal)

    print(f"Finished {area} setup")

    return traindata, valdata, testdata