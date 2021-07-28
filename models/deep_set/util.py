import pickle

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import MultiSetSequence

# General Utility Functions used in Model Training and For HyperParameter Optimisation --> factored out to have clearer HP optim file

def get_mask(sizes, max_size):
    return (torch.arange(max_size).reshape(1, -1).to(sizes.device) < sizes.unsqueeze(2))


def get_dataset(num_pixels, max_set_len,gal, path_to_data='../../bricks_data/multiset.pickle'):
    with open(path_to_data, 'rb') as f:
        mini_multiset = pickle.load(f)
        f.close()
    df = pd.DataFrame.from_dict(mini_multiset, orient='index')
    train_df, test_df = train_test_split(df, test_size=0.33, random_state=44, shuffle=True)
    traindata = MultiSetSequence(dict=train_df.to_dict(orient='index'), num_pixels=round(num_pixels * 0.67),
                                 max_ccds=max_set_len)
    traindata.set_targets(gal_type=gal)
    testdata = MultiSetSequence(dict=test_df.to_dict(orient='index'), num_pixels=round(num_pixels * 0.33),
                                max_ccds=max_set_len)
    testdata.set_targets(gal_type=gal)


    return traindata, testdata