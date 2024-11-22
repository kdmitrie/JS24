from abc import abstractmethod, ABC

import polars as pl
import numpy as np


class JSModel(ABC):
    def __init__(self):
        time_sin_cos = ['time_sin', 'time_cos']
        features = [f'feature_{n:02d}' for n in range(79)]

        self.today_cols = time_sin_cos + features

    def reinit(self):
        pass

    def set_cols(self, data_cols, lags_cols) -> None:
        self.data_cols = data_cols
        self.lags_cols = lags_cols

        self.data_cols_today = [self.data_cols.index(col) for col in self.today_cols]
        self.data_cols_weight = self.data_cols.index('weight')

        self.lags_cols_today = [self.lags_cols.index(col) for col in self.today_cols]
        self.lags_cols_weight = self.lags_cols.index('weight')
        self.lags_cols_target = self.lags_cols.index('responder_6_lag_1')

    def apply_yesterday_data(self, yesterday_data: np.ndarray) -> None:
        """
        yesterday_data [Symbols x Timesteps x self.lags_cols]
        """
        self.yesterday_data = yesterday_data

    def predict(self, data: np.ndarray) -> pl.DataFrame:
        """
        data [Symbols x self.data_cols]
        """
        predictions = data[~np.isnan(data[:, 0]), :2]
        predictions[:, 1] = 0
        predictions = pl.DataFrame(predictions, schema={'row_id': pl.Int64, 'responder_6': pl.Float32})
        return predictions

    @abstractmethod
    def save(self, name: str) -> None: pass

    @abstractmethod
    def load(self, name: str) -> None: pass