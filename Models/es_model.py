from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import torch
from config import *
import pandas as pd
import numpy as np
from torch.utils.dlpack import to_dlpack


class ES_series():
    def __init__(self, time_series, seasons, n_output=OUTPUT_SIZE, alpha=None, beta=None, gamma=None):
        self.time_series = time_series
        self.seasons = seasons
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.forecast_horzion = n_output
        self.no_series = len(time_series.columns)

    def forecast_data(self):
        res = pd.Series([])
        levels = pd.Series([])
        seasons = pd.Series([])
        if self.no_series > MAX_NUM_SERIES:
            sample_sz = MAX_NUM_SERIES
        else:
            sample_sz = MIN_NUM_SERIES
        for i in range(sample_sz):
            curr_series = self.time_series.iloc[:MAX_SERIES_LEN, i]
            x = curr_series.dropna().values
            tmp_mod = ExponentialSmoothing(x, seasonal_periods=self.seasons)
            decomp = seasonal_decompose(x, model='multiplicative', freq=self.seasons)
            tmp_fit = tmp_mod.fit(smoothing_level=self.alpha, smoothing_slope=self.beta, smoothing_seasonal=self.gamma)
            tmp_pred = tmp_fit.fittedvalues
            levels = pd.concat([levels, pd.Series(tmp_fit.level)], axis=1)
            seasons = pd.concat([seasons, pd.Series(decomp.seasonal)], axis=1)
            res = pd.concat([res, pd.Series(tmp_pred)], axis=1)
        return res.iloc[:, 1:], levels.iloc[:, 1:], seasons.iloc[:, 1:]







