import pandas as pd
import numpy as np
import os
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

class ARIMAGridSearch:
    def __init__(self, p_range, d_range, q_range):
        self.p_range = p_range
        self.d_range = d_range
        self.q_range = q_range
    
    def search(self, time_series):
        best_aic = np.inf
        best_order = None
        best_model = None
        
        for order in product(self.p_range, self.d_range, self.q_range):
            try:
                model = ARIMA(time_series, order=order)
                model_fit = model.fit()
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = order
                    best_model = model_fit
            except Exception as e:
                continue
        return best_order, best_aic, best_model