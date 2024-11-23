import numpy as np
import pickle
import polars as pl
from .base import JSModel


class JSModel_SimpleMLP(JSModel):
    replace_nan = 0

    def __init__(self, tf_model, features_mean, features_std):
        super().__init__()
        self.tf_model = tf_model
        self.features_mean = features_mean
        self.features_std = features_std

    def set_cols(self, data_cols, lags_cols) -> None:
        super().set_cols(data_cols, lags_cols)
        self.features_cols = [self.data_cols.index(f'feature_{n:02d}') for n in range(79)]

    def model_predict(self, X):
        model_X = X.reshape(X.shape[1:])
        model_X = model_X / self.features_mean
        y = self.tf_model.predict(model_X, verbose=0)[None, ...]
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
        self.tf_model.save_weights(f'{name}.weights.h5')
        with open(f'{name}.mean_std.pkl', 'wb') as f:
            pickle.dump((self.features_mean, self.features_std), f)

    def load(self, name: str) -> None:
        self.tf_model.load_weights(f'{name}.weights.h5')
        with open(f'{name}.mean_std.pkl', 'rb') as f:
            self.features_mean, self.features_std = pickle.load(f)
