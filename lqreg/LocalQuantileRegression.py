# Author: Keith O'Hara
# License: Apache Version 2.0

from typing import Union
import numpy as np
import pandas as pd

from sklearn.linear_model import QuantileRegressor
from statsmodels.regression.quantile_regression import QuantReg

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

class LocalQuantileRegression:
    '''
    Local Quantile Regression
    '''
    def __init__(
        self,
        target: Union[pd.Series, pd.DataFrame, np.ndarray],
        features: Union[pd.Series, pd.DataFrame, np.ndarray]
    ):
        self.n = target.shape[0]

        if len(features.shape) == 1:
            self.K = 1
        else:
            self.K = features.shape[1]

        if isinstance(target, (pd.Series, pd.DataFrame)):
            self.Y = target.to_numpy()
        elif isinstance(target, np.ndarray):
            self.Y = target
        else:
            raise Exception("The 'target' vector must be of type 'pandas.DataFrame', 'pandas.Series', or 'numpy.ndarray'")
        
        if isinstance(features, (pd.Series, pd.DataFrame)):
            self.X = features.to_numpy()
        elif isinstance(features, np.ndarray):
            self.X = features
        else:
            raise Exception("The 'features' vector/matrix must be of type 'pandas.DataFrame', 'pandas.Series', or 'numpy.ndarray'")

        if self.K == 1:
            self.X = self.X[:, np.newaxis]
    
    @staticmethod
    def _dsnorm(x, log_form = False):
        if log_form:
            return - 0.5 * (np.log(2.0 * np.pi) + (x * x))
        else:
            return np.exp(- x * x / 2.0) / np.sqrt(2.0 * np.pi)

    def _dsmvnorm(self, X):
        ret_val = 0.0

        for k in range(self.K):
            ret_val = ret_val + self._dsnorm(X[:,k], True)
        
        return np.exp(ret_val)

    def _QRFit(
        self,
        local_features,
        weights_vec,
        tau,
        fit_method
    ):
        if fit_method == "sk":
            rq = QuantileRegressor(quantile = tau, alpha = 0.0, fit_intercept = False, solver = 'highs')
            rq_res = rq.fit(np.column_stack((np.ones(self.n), local_features)), self.Y, weights_vec)
            return rq_res.coef_[0]
        elif fit_method == "sm":
            rq = QuantReg(weights_vec * self.Y, np.column_stack((weights_vec, local_features * weights_vec[:, np.newaxis])))
            rq_res = rq.fit(q = tau, vcov = 'iid', max_iter = 2000)
            return rq_res.params[0]
        else:
            raise Exception("fit_method must be in 'sk', 'sm'")
        
    def fit(
        self,
        tau: float = 0.5,
        bandwidth: float = 1.0,
        fit_method: str = "sk"
    ) -> np.array:
        '''
        Fit method for LocalQuantileRegression class
        '''
        local_fit = np.zeros(self.n)

        for i in range(self.n):
            local_features = self.X - self.X[i,:]
            weights_vec = self._dsmvnorm(local_features / bandwidth)
            local_fit[i] = self._QRFit(local_features, weights_vec, tau, fit_method)

        return local_fit
