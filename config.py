import torch
# Exponential Smoothing Parameters

# Convolution Layers Parameters
KERNEL_SIZE = 5 #[3 5 7 9]
STRIDE = 1
POOL = 2

# General Model Parameters
INPUT_SIZE = 52
BATCH_SIZE = 4
NO_CLASS = 6
MAX_NUM_SERIES = 500
MIN_NUM_SERIES = 13
OUTPUT_SIZE = 14# Should be equal to the forecasting horizon
EPOCHS = 23
SEASONS = 52
USE_CUDA = False #torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")
MIN_IN_SEQ_LEN = 0
MIN_SERIES_LEN = OUTPUT_SIZE + INPUT_SIZE + MIN_IN_SEQ_LEN + 2
MAX_SERIES_LEN = 6*SEASONS + MIN_SERIES_LEN

# LSTM Parameters
LSTM_HIDDEN = 32
NUM_LAYERS = 9
DILATION = {'Weekly': [1, 52],
            'Monthly': [1, 3, 6, 12],
            'Quarterly': [[1, 2], [4, 8]],
            'Yearly': [1, 6]}
CELL_TYPE = 'QRNN'

# Directory Information
CWD_PATH = "C:/Users/Vincent/PycharmProjects/ECE_542_Final_Project/"
DATA_PATH = "C:/Users/Vincent/PycharmProjects/ECE_542_Final_Project/Data/"
TRAIN_PATH = "C:/Users/Vincent/PycharmProjects/ECE_542_Final_Project/Data/Train/Weekly-train.csv"
TEST_PATH = "C:/Users/Vincent/PycharmProjects/ECE_542_Final_Project/Data/Test/Weekly-test.csv"
MODEL_PATH = "C:/Users/Vincent/PycharmProjects/ECE_542_Final_Project/Models"