"""Factored out utility functions to load and preprocess datasets."""

import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import DensitySurvey


def get_dataset(gal='lrg', num_pixels=None, kit=False):
    if kit:
        df = pd.read_csv('../../bricks_data/dataset_kitanidis.csv')
    else:
        df = pd.read_csv('../../bricks_data/dataset_geometric.csv')

    # ToDo: At later stage you can pass a list of pixel indeces to filter test and train sets

    if num_pixels is not None:
        df = df.sample(n=num_pixels, replace=False, random_state=44, axis=0)

    df.drop('pixel_id', axis=1, inplace=True)

    train_df, test_df = train_test_split(df, test_size=0.33, random_state=44, shuffle=True)
    traindata = DensitySurvey(train_df, gal)
    scaler_in, scaler_out = traindata.__getscaler__()
    testdata = DensitySurvey(test_df, gal, scaler_in, scaler_out)

    return traindata, testdata


def get_full_dataset(num_pixels=None, area='des', gal='lrg'):
    df_train = pd.read_csv(f'data/{area}/{area}.csv')
    df_train = df_train.drop(columns=['pixel_id', 'exposures'], axis=1, inplace=False)

    if num_pixels is not None:
        df_train = df_train.sample(n=num_pixels, replace=False, random_state=666, axis=0)
    df_test = pd.read_csv(f'data/{area}/{area}_test.csv')
    df_test = df_test.drop(columns=['pixel_id', 'exposures'], axis=1, inplace=False)

    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=666, shuffle=True)

    traindata = DensitySurvey(df_train, gal)
    valdata = DensitySurvey(df_val, gal)
    testdata = DensitySurvey(df_test, gal)

    return traindata, valdata, testdata
