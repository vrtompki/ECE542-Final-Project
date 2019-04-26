from torch.utils import data
import torch
import numpy as np
import pandas as pd
from config import *
from data_prep import load_smooth_data


class TS_Dataset(data.Dataset):
    def __init__(self, mode='train', transform=None):
        self.transform = transform
        self.mode = mode
        self.train_dict, self.test_dict = load_smooth_data()
        if self.mode == 'train':
            self.smoothed = self.train_dict['smoothed']
            self.levels = self.train_dict['levels']
            self.seasonality = self.train_dict['seasonality']
            self.data_len = MAX_SERIES_LEN
            self.orig_series = np.transpose(pd.read_csv(TRAIN_PATH, header=0, index_col=0).dropna().values)
        else:
            self.smoothed = self.train_dict['smoothed']
            self.levels = self.train_dict['levels']
            self.seasonality = self.train_dict['seasonality']
            self.data_len = MIN_NUM_SERIES
            self.orig_series = np.transpose(pd.read_csv(TEST_PATH, header=0, index_col=0).dropna().values)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # Select sample
        dat = self.smoothed.values
        dat = dat.reshape((dat.shape[0], dat.shape[1], 1))
        input_tensor = torch.Tensor(dat[index])
        label_tensor = torch.Tensor(self.orig_series[index])
        return input_tensor, label_tensor
