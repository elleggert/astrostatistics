"""Factored out utility functions to load and preprocess datasets."""

import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import DensitySurvey

def get_full_dataset(num_pixels=None, area='des', gal='lrg'):
    df_train = pd.read_csv(f'data/{area}/{area}_512.csv')
    df_train = df_train.drop(columns=['pixel_id', 'exposures'], axis=1, inplace=False)

    if num_pixels is not None:
        df_train = df_train.sample(n=num_pixels, replace=False, random_state=666, axis=0)
    df_test = pd.read_csv(f'data/{area}/{area}_test_512.csv')
    df_test = df_test.drop(columns=['pixel_id', 'exposures'], axis=1, inplace=False)

    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=666, shuffle=True)

    traindata = DensitySurvey(df_train, gal)
    valdata = DensitySurvey(df_val, gal)
    testdata = DensitySurvey(df_test, gal)

    return traindata, valdata, testdata



def get_final_dataset(num_pixels=None, area='des', gal='lrg'):
    df_train = pd.read_csv(f'data/{area}/{area}_512.csv')
    df_train = df_train.drop(columns=['pixel_id', 'exposures'], axis=1, inplace=False)

    if num_pixels is not None:
        df_train = df_train.sample(n=num_pixels, replace=False, random_state=666, axis=0)
    df_test = pd.read_csv(f'data/{area}/{area}_test_512.csv')
    df_test = df_test.drop(columns=['pixel_id', 'exposures'], axis=1, inplace=False)

    traindata = DensitySurvey(df_train, gal)
    testdata = DensitySurvey(df_test, gal)

    return traindata, testdata, testdata