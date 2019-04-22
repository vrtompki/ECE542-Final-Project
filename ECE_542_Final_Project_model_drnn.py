from Models.drnn import DRNN
from Models import es_model

train_epochs = 23
seasonality = 52
n_input = 10
n_hidden = 32
n_output = 13
level_variability_penalty = 100
c_state_penalty = 0
per_series_lr_mult = 1
min_inp_seq_len = 0
min_series_len = n_output + n_input + min_inp_seq_len + 2
max_series_len = 6*seasonality + min_series_len
n_layers = 9
cell_type = 'QRNN'
topn = 3
init_learning_rate = 1e-3

# Load and process data

smooth_model = es_model.ES_series(seasonality=seasonality)
smooth_input = smooth_model.forecast_data()

model = DRNN(n_input, n_hidden, n_layers, cell_type, dilation=[1, 52])
pred_res = model(smooth_input)



