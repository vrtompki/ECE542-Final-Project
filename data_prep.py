from config import *
from torch.utils.data import DataLoader
from Models import es_model
import numpy as np
import pandas as pd
import multiprocessing
import torch


test_path = TEST_PATH
train_path = TRAIN_PATH


def load_smooth_data():
    train_dict ={}
    test_dict ={}
    train_data = pd.read_csv(TRAIN_PATH, header=0, index_col=0)
    test_data = pd.read_csv(TEST_PATH, header=0, index_col=0)
    smooth_train = es_model.ES_series(train_data, seasons=SEASONS)
    smooth_test = es_model.ES_series(test_data, seasons=SEASONS)
    smooth_train, train_levels, train_seasonality = smooth_train.forecast_data()
    smooth_test, test_levels, test_seasonality = smooth_test.forecast_data()
    train_dict['smoothed'] = smooth_train
    train_dict['levels'] = train_levels
    train_dict['seasonality'] = train_seasonality
    test_dict['smoothed'] = smooth_test
    test_dict['levels'] = test_levels
    test_dict['seasonality'] = test_seasonality
    return train_dict, test_dict




