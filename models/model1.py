import numpy as np
import polars as pl
from .base import JSModel
import tensorflow as tf


def r2_loss(y_true, y_pred, weight):
    return sum(weight * (y_true - y_pred) ** 2) / sum(weight * y_true ** 2)


class JSModel_1(JSModel):
    replace_nan = 3

    def __init__(self):
        super().__init__()
        self.tf_model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(50, 81)),
            tf.keras.layers.Dense(1,
                                  activation='tanh',
                                  kernel_initializer='zeros',
                                  bias_initializer='zeros',
                                  ),
        ])
        # self.tf_model.compile(optimizer='adam', loss=tf.keras.losses.MSE, metrics=[])
        self.optimizer = tf.optimizers.Adam(learning_rate=1e-4)
        self.loss = r2_loss  # tf.keras.losses.MSE

    def apply_yesterday_data(self, yesterday_data: np.ndarray) -> None:
        """
        yesterday_data [Symbols x Timesteps x self.lags_cols]
        """
        self.yesterday_data = yesterday_data

        # Get yesterday data for training
        X = yesterday_data[:, :, self.lags_cols_today]
        y = yesterday_data[:, :, self.lags_cols_target]
        weight = yesterday_data[:, :, self.lags_cols_weight]

        # Reshapa training data and exclude NaNs
        y1 = y.reshape((-1,))
        mask = ~np.isnan(y1)
        X1 = X.reshape((-1, X.shape[-1]))[mask]
        weight1 = weight.reshape((-1,))[mask]
        y1 = y1[mask]

        X1 = np.nan_to_num(X1, self.replace_nan)
        X1 = X1[:, None, :]

        with tf.GradientTape() as tape:
            pred = self.model_predict(X1)[:, 0, 0]
            loss = self.loss(y1, pred, weight1)

            print(f'\tyesterday loss = {loss.numpy():.4f}')

        grads = tape.gradient(loss, self.tf_model.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.tf_model.trainable_variables))

    def model_predict(self, X):
        y = self.tf_model(X)
        return 5 * y

    def predict(self, data: np.ndarray) -> pl.DataFrame:
        """
        data [Symbols x self.data_cols]
        """

        X = np.nan_to_num(data[None, :, self.data_cols_today], self.replace_nan)
        y = self.model_predict(X)

        row_ids = data[:, 0: 1]
        predictions = np.hstack((row_ids, y[0, :, 0: 1]))
        predictions = predictions[~np.isnan(data[:, 0]), :]

        predictions = pl.DataFrame(predictions, schema={'row_id': pl.Int64, 'responder_6': pl.Float32})
        return predictions

    def save(self, name: str) -> None:
        self.tf_model.save_weights(f'{name}.weights.h5')

    def load(self, name: str) -> None:
        self.tf_model.load_weights(f'{name}.weights.h5')
