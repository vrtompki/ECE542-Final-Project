from statsmodels.tsa.holtwinters import ExponentialSmoothing

import cupy
import torch
import pandas as pd
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

class ES_series():
    def __init__(self, time_series, seasonality, n_output=13, alpha=None, beta=None, gamma=None):
        self.time_series = time_series
        self.seasonality = seasonality
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.forecast_horzion = n_output

    def forecast_data(self):
        res = pd.Series([])
        for i in range(len(self.time_series)):
            x = self.time_series[i,:]
            es_mod = ExponentialSmoothing(x)
            fit_mod = es_mod.fit(smoothing_level=self.alpha, smoothing_slope=self.beta,
                                 smoothing_seasonal=self.gamma)
            tmp_res = es_mod.predict(es_mod.params)
            res = res.add(tmp_res)
        return res







