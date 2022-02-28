"""Define the Datasets used for DeepSets"""

import pickle
from torch.utils.data import DataLoader, Dataset
import torch
import random
import pandas as pd
from sklearn import preprocessing
import numpy as np


# noinspection PyAttributeOutsideInit
class SetSequence(Dataset):
    """Processes and Returns a Dataset of Variable Sized Input Sets of Dimensions
    N = Number Pixels of that are returned
    M = Max Size of each Individual Set of CCDs


    """

    def __init__(self, num_pixels=10, max_ccds=30, var_set_len=False):

        with open('../../bricks_data/pixel2ccd_256_non_inclusive.pickle', 'rb') as f:
            self.pixel2ccd_dict = pickle.load(f)
            f.close()

        self.ccd = CCD()
        self.num_features = self.ccd.num_features

        # Dimensions
        self.num_pixels = num_pixels
        self.max_ccds = max_ccds
        self.var_set_len = var_set_len

        df_raw = pd.read_csv('../../bricks_data/dataset_geometric.csv')
        # Randomly Sampling Pixel Indices from Dataframe
        pixel_indices = random.sample(range(len(df_raw)), num_pixels)

        self.df = df_raw.iloc[pixel_indices]
        self.pix_ids = self.df.pixel_id.to_numpy()

        self.initialise_inputs()

        self.initialise_lengths()

        # Target
        self.label = np.random.rand(self.num_pixels * self.max_ccds)

        # Mask Variable Len Sets
        # self.set_max_set_len()

    def set_targets(self, gal_type):
        # Features and inputs:
        self.target = None
        self.target = self.df[gal_type].to_numpy()
        # print(self.target.shape)
        self.scaler_out = preprocessing.MinMaxScaler()
        self.target = self.scaler_out.fit_transform(self.target.reshape(-1, 1))
        # print(self.target.shape)

    def initialise_lengths(self):
        self.lengths = np.zeros(self.num_pixels, dtype=int)
        if self.var_set_len:
            for i, pix in enumerate(self.pix_ids):
                c = len(self.pixel2ccd_dict[pix])
                if c < self.max_ccds:
                    self.lengths[i] = c
                else:
                    self.lengths[i] = self.max_ccds

        else:
            self.lengths.fill(self.max_ccds)

    def initialise_inputs(self):
        # self.input = -1 * np.ones((self.num_pixels, self.max_ccds, self.num_features))
        self.input = np.zeros((self.num_pixels, self.max_ccds, self.num_features))

        # Iterate through the pixels
        for i, pix in enumerate(self.pix_ids):
            ids = self.pixel2ccd_dict[pix]
            random.shuffle(ids)
            # print(len(ids))
            ids = ids[:self.max_ccds]
            # print(len(ids))
            # print()
            x = self.ccd.get_ccds(ids)
            # Iterate through the CCDs for every pixel
            for j in range(len(ids)):
                self.input[i, j] = x[j]

    def set_max_set_len(self):
        self.index_matrix = -1 * np.ones((self.num_pixels, self.max_ccds), dtype=int)

        # Getting random labels for now, in the future this will be the output densities

        m = 0
        for i in range(self.num_pixels):

            for j in range(self.lengths[i]):
                ''' This code with label == 0 is not yet needed, but this masking will become necessary when I have
                    I have 64 subpixels per pixel and some of those are not covered by CCDs'''
                while self.label[m] == 0:
                    m += 1
                self.index_matrix[i, j] = m
                m += 1

        print(self.lengths)
        print(self.index_matrix)

    def __len__(self):
        return self.num_pixels

    def __getitem__(self, idx):
        x = torch.from_numpy(self.input[idx]).float()
        # x = x.unsqueeze(0)
        y = torch.tensor(self.target[idx, 0]).float()
        # print(y.shape)
        y = y.unsqueeze(-1)
        # print(y.shape)

        # l = torch.tensor(self.lengths[idx])
        l = self.lengths[idx]

        return x, y, l


# ToDo: Adapt to stack stage2inputs earlier and make that more efficient.


# noinspection PyAttributeOutsideInit
class MultiSetSequence(Dataset):
    """Processes and Returns a Dataset of Variable Sized Input Sets of Dimensions
    N = Number SubPixels of that are returned --> usually 64
    M = Max Size of each Individual Set of CCDs
    """

    def __init__(self, dict, num_pixels, max_ccds, num_features, num_subpixels=16, test=False):
        self.mini_multiset = dict
        if num_pixels < len(self.mini_multiset):
            self.num_pixels = num_pixels
        else:
            self.num_pixels = len(self.mini_multiset)

        # Initialise DataSet
        self.num_features = num_features
        self.input = np.zeros((self.num_pixels, num_subpixels, max_ccds, num_features))
        self.lengths = np.zeros((self.num_pixels, num_subpixels), dtype=int)
        self.lrg = np.zeros(self.num_pixels)
        self.elg = np.zeros(self.num_pixels)
        self.qso = np.zeros(self.num_pixels)
        self.glbg = np.zeros(self.num_pixels)
        self.rlbg = np.zeros(self.num_pixels)
        self.hinh = np.zeros(self.num_pixels)
        self.stellar = np.zeros(self.num_pixels)
        self.gaia = np.zeros(self.num_pixels)
        self.gaia12 = np.zeros(self.num_pixels)
        self.sagitarius = np.zeros(self.num_pixels)
        self.ebv = np.zeros(self.num_pixels)
        self.test = test
        if self.test is True:
            self.pixel_id = np.zeros(self.num_pixels)

        self.initialise_inputs()

        self.clean_nans()

    def set_targets(self, gal_type, scaler=None):
        self.scaler = scaler
        # Features and inputs:
        self.target = None
        if gal_type == 'lrg':
            self.target = self.lrg
        if gal_type == 'elg':
            self.target = self.elg
        if gal_type == 'qso':
            self.target = self.qso
        if gal_type == 'glbg':
            self.target = self.glbg
        if gal_type == 'rlbg':
            self.target = self.rlbg
        self.target = self.target.reshape(-1, 1)
        """if self.scaler is None:
            self.scaler = preprocessing.MinMaxScaler()
        self.target = self.scaler.fit_transform(self.target.reshape(-1, 1))"""

    def initialise_inputs(self):
        for i, pix in enumerate(self.mini_multiset):
            if i >= self.num_pixels:
                break

            # For refactoring, morph in the data preprocessing already --> this is super messy
            self.input[i] = self.mini_multiset[pix][0]
            self.lengths[i] = self.mini_multiset[pix][1]
            self.lrg[i] = self.mini_multiset[pix][2]
            self.elg[i] = self.mini_multiset[pix][3]
            self.qso[i] = self.mini_multiset[pix][4]
            self.glbg[i] = self.mini_multiset[pix][5]
            self.rlbg[i] = self.mini_multiset[pix][6]
            self.stellar[i] = self.mini_multiset[pix][7]
            self.hinh[i] = self.mini_multiset[pix][8]
            self.gaia[i] = self.mini_multiset[pix][9]
            self.gaia12[i] = self.mini_multiset[pix][10]
            self.sagitarius[i] = self.mini_multiset[pix][11]
            self.ebv[i] = self.mini_multiset[pix][12]
            if self.test is True:
                self.pixel_id[i] = self.mini_multiset[pix][13]

    def __len__(self):
        return self.num_pixels

    def __getitem__(self, idx):
        x1 = torch.from_numpy(self.input[idx]).float()
        x2 = torch.from_numpy(self.stage2_input[idx]).float()
        # x = x.unsqueeze(0)
        y = torch.tensor(self.target[idx, 0]).float()
        # print(y.shape)
        y = y.unsqueeze(-1)
        # print(y.shape)

        # l = torch.tensor(self.lengths[idx])
        l = self.lengths[idx]

        return x1, x2, y, l

    def clean_nans(self):
        nan_list = []
        for i in range(self.num_pixels):
            cond = np.isnan(self.input[i])
            if cond.sum():
                nan_list.append(i)

        self.input = np.delete(self.input, nan_list, axis=0)
        self.lengths = np.delete(self.lengths, nan_list, axis=0)
        self.lrg = np.delete(self.lrg, nan_list, axis=0)
        self.elg = np.delete(self.elg, nan_list, axis=0)
        self.qso = np.delete(self.qso, nan_list, axis=0)
        self.glbg = np.delete(self.glbg, nan_list, axis=0)
        self.rlbg = np.delete(self.rlbg, nan_list, axis=0)
        self.stellar = np.delete(self.stellar, nan_list, axis=0)
        self.hinh = np.delete(self.hinh, nan_list, axis=0)
        self.gaia = np.delete(self.gaia, nan_list, axis=0)
        self.gaia12 = np.delete(self.gaia12, nan_list, axis=0)
        self.sagitarius = np.delete(self.sagitarius, nan_list, axis=0)
        self.ebv = np.delete(self.ebv, nan_list, axis=0)
        if self.test is True:
            self.pixel_id = np.delete(self.pixel_id, nan_list, axis=0)

        self.stage2_input = np.stack((self.stellar, self.hinh, self.gaia, self.gaia12, self.sagitarius, self.ebv),
                                     axis=1)
        self.num_pixels = len(self.lrg)
