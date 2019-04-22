# Exponential Smoothing Parameters

# Convolution Layers Parameters
KERNEL_SIZE = 5 #[3 5 7 9]
STRIDE = 1
POOL = 2

# General Model Paramters
INPUT_SIZE = 150
BATCH_SIZE = 32
NO_CLASS = 2
OUTPUT_SIZE = 14# Should be equal to the forecasting horizon

# LSTM Parameters
LSTM_HIDDEN = 20
NUM_LAYERS = 4
DILATION = {'Weekly': [1, 52],
              'Monthly': [1, 3, 6, 12],
              'Quarterly': [[1, 2], [4, 8]],
              'Yearly': [1, 6]}

# Directory Information
CWD_PATH = "C:/Users/Vincent/PycharmProjects/ECE_542_Final_Project/"
DATA_PATH = "C:/Users/Vincent/PycharmProjects/ECE_542_Final_Project/Data/"
TRAIN_PATH = "C:/Users/Vincent/PycharmProjects/ECE_542_Final_Project/Data/Train"
TEST_PATH = "C:/Users/Vincent/PycharmProjects/ECE_542_Final_Project/Data/Test"
MODEL_PATH = "C:/Users/Vincent/PycharmProjects/ECE_542_Final_Project/Models"