import numpy as np
import polars as pl
import pickle
from .base import JSModel


class JSModel_Ridge(JSModel):
    replace_nan = 3

    def __init__(self, sk_model):
        super().__init__()
        self.sk_model = sk_model

    def set_cols(self, data_cols, lags_cols) -> None:
        super().set_cols(data_cols, lags_cols)
        self.features_cols = [self.data_cols.index(f'feature_{n:02d}') for n in range(79)]

    def model_predict(self, X):
        y = self.sk_model.predict(X.reshape(X.shape[1:]))[None, ...]
        return y

    def predict(self, data: np.ndarray) -> pl.DataFrame:
        """
        data [Symbols x self.data_cols]
        """

        X = np.nan_to_num(data[None, :, self.features_cols], self.replace_nan)
        y = self.model_predict(X)

        row_ids = data[:, 0: 1]
        predictions = np.hstack((row_ids, y[0, :, 0: 1]))
        predictions = predictions[~np.isnan(data[:, 0]), :]

        predictions = pl.DataFrame(predictions, schema={'row_id': pl.Int64, 'responder_6': pl.Float32})
        return predictions

    def save(self, name: str) -> None:
        with open(f'{name}.pkl', 'wb') as f:
            pickle.dump(self.sk_model, f)

    def load(self, name: str) -> None:
        with open(f'{name}.pkl', 'rb') as f:
            self.sk_model = pickle.load(f)
