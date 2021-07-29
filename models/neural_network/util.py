import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import DensitySurvey


def get_dataset(gal ='lrg', num_pixels=None, kit=False):

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