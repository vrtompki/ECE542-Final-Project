from config import DATA_PATH, TEST_PATH, TRAIN_PATH
import numpy as np
import pandas as pd

test_path = TEST_PATH+'/Weekly-test.csv'
train_path = TRAIN_PATH +'/Weekly-train.csv'


def load_data():
    train_data = pd.read_csv(train_path, header=0, index_col=0)
    train_values = train_data.values
    test_data = pd.read_csv(test_path, header=0, index_col=0)
    test_values = test_data.values
    return train_values, test_values


