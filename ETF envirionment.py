import warnings
warnings.filterwarnings('ignore')

from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model

import numpy as np
import pandas as pd
import os

import random
from collections import deque
import matplotlib.pylab as plt

from sklearn.decomposition import PCA

class ETFEnvironment:
    
    def __init__(self, volumes = '/content/drive/My Drive/volumes (3).txt',
                       prices = '/content/drive/My Drive/prices5.txt',
                       returns = '/content/drive/My Drive/returns7.txt', 
                       capital = 1e6):
        
        self.returns = returns
        self.prices = prices
        self.volumes = volumes   
        self.capital = capital  
        
        self.data = self.load_data()

    def load_data(self):
        volumes = np.genfromtxt(self.volumes, delimiter=',')[2:, 1:]
        prices = np.genfromtxt(self.prices, delimiter=',')[2:, 1:]
        returns=pd.read_csv(self.returns, index_col=0)
        assets=np.array(returns.columns)
        dates=np.array(returns.index)
        returns=returns.to_numpy()
        return pd.DataFrame(prices, 
             columns = assets,
             index = dates
            )
    
    def preprocess_state(self, state):
        return state
    
    def get_state(self, t, lookback, is_cov_matrix = True, is_raw_time_series = False):
        assert lookback <= t
        decision_making_state = self.data.iloc[t-lookback:t]
        decision_making_state = decision_making_state.pct_change().dropna()
        if is_cov_matrix:
            x = decision_making_state.cov()
            return x
        else:
            if is_raw_time_series:
                decision_making_state = self.data.iloc[t-lookback:t]
            return self.preprocess_state(decision_making_state)

    def get_reward(self, action, action_t, reward_t):
        
        def local_portfolio(returns, weights):
            weights = np.array(weights)
            rets = returns.mean() # * 252
            covs = returns.cov() # * 252
            P_ret = np.sum(rets * weights)
            P_vol = np.sqrt(np.dot(weights.T, np.dot(covs, weights)))
            P_sharpe = P_ret / P_vol
            return np.array([P_ret, P_vol, P_sharpe])
        
        weights = action
        returns = self.data[action_t:reward_t].pct_change().dropna()
        
        rew = local_portfolio(returns, weights)[-1]
        rew = np.array([rew] * len(self.data.columns))
        
        return np.dot(returns, weights), rew
