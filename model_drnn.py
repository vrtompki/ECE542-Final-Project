from Models.drnn import DRNN
import multiprocessing
import pandas as pd
import sys
from torch import optim
from tqdm import tqdm
from dataset_utils import TS_Dataset
import torch
from torch.utils.data import DataLoader
from config import *

n_input = 4
n_hidden = 32
n_output = 13
level_variability_penalty = 100
c_state_penalty = 0
per_series_lr_mult = 1


topn = 3
init_learning_rate = 1e-3

train_acc = []
train_loss = []


def main():
    training_set = TS_Dataset()
    data_loaders = DataLoader(training_set, batch_size=BATCH_SIZE*MAX_SERIES_LEN, shuffle=False, drop_last=True,
                                       num_workers=multiprocessing.cpu_count())
    model = DRNN(n_input=INPUT_SIZE, n_hidden=LSTM_HIDDEN, n_layers=NUM_LAYERS, cell_type=CELL_TYPE, dilation=[1, 52],
                 batch_first=True)
    opt = optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = data_loaders
    for epoch in range(EPOCHS):
        print("\nEpoch " + str(epoch + 1))
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for local_batch, local_labels in tqdm(train_loader, file=sys.stdout):
            local_batch, local_labels = local_batch.to(DEVICE), local_labels.to(DEVICE)

            forecast_preds = model(local_batch)
            loss = criterion(forecast_preds, local_labels)
            loss.backward()
            opt.step()
            opt.zero_grad()
            series_preds = torch.argmax(forecast_preds.data, 1)
        # losses, nums, corrects = loss_batch(model, criterion, image, label, opt)
        train_loss.append(running_loss / len(train_loader.dataset))
        train_acc.append(running_corrects.item() / (len(train_loader.dataset)))
        print("Training loss:", train_loss[-1])
        print("Training accuracy: %.4f" % train_acc[-1])

# Load and process data
if __name__ == '__main__':
    main()
# for i in range(len(smooth_input)):
#     pred_res = model(torch.Tensor(smooth_input[i,:]))
#     result_series.add(pred_res)



