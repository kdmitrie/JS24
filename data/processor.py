from dataclasses import dataclass
import polars as pl
import numpy as np
from ..models.base import JSModel


@dataclass
class JSCommonProcessor:
    model: JSModel
    num_symbols: int = 50
    default_num_timesteps: int = 849

    def __post_init__(self) -> None:
        self._smb_ts_cache = {}
        self._day_data = None
        self.num_timesteps = self.default_num_timesteps
        self.df_symbols = pl.DataFrame({'symbol_id': range(self.num_symbols)}, schema={'symbol_id': pl.Int8})
        self.data_cols = []
        self.lags_cols = []

    def _get_smb_ts_df(self, num_timesteps: int) -> pl.DataFrame:
        """
            Generate a dataframe with all possible values of symbols and timesteps
        """

        if num_timesteps in self._smb_ts_cache:
            return self._smb_ts_cache[num_timesteps]

        time_id = np.arange(num_timesteps)
        time_sin = np.sin(2 * np.pi / num_timesteps * time_id)
        time_cos = np.cos(2 * np.pi / num_timesteps * time_id)
        df_timesteps = pl.DataFrame({'time_id': time_id, 'time_sin': time_sin, 'time_cos': time_cos},
                                    schema={'time_id': pl.Int16, 'time_sin': pl.Float32, 'time_cos': pl.Float32})
        # self.df_symbols = pl.DataFrame({'symbol_id': range(self.num_symbols)}, schema={'symbol_id': pl.Int8})
        self._smb_ts_cache[num_timesteps] = self.df_symbols.join(df_timesteps, how='cross')
        return self._smb_ts_cache[num_timesteps]

    def _day_start(self) -> None:
        """
        Create a room for accumulating all the data during the day
        """

        self._day_data = []

    def _day_accumulate(self, batch: pl.DataFrame, predictions: pl.DataFrame) -> None:
        """
        Accumulate a single timestep data, including the provided features and model output
        """

        accumulated = batch.join(predictions, on='row_id')
        self._day_data.append(accumulated)

    def _day_end(self, lags: pl.DataFrame) -> np.ndarray:
        """
        Join all accumulated data togeteher and add lagged data

        Returns: yesterday_data [Symbols x Timesteps x Features]
        """

        self.num_timesteps = lags.select('time_id').max().item() + 1
        empty = self._get_smb_ts_df(self.num_timesteps)

        yesterday_data_with_blanks = pl. \
            concat(self._day_data). \
            join(lags, on=['date_id', 'time_id', 'symbol_id'])

        # Create 3d numpy array with NaNs
        yesterday_data = empty.join(yesterday_data_with_blanks, on=['time_id', 'symbol_id'], how='left')
        yesterday_data = yesterday_data.select(self.lags_cols)
        yesterday_data = yesterday_data.to_numpy().reshape((self.num_symbols, self.num_timesteps, -1))

        return yesterday_data

    def set_cols(self, data_cols, lags_cols) -> None:
        """
        Set the columns for item and lags data
        """
        self.data_cols = data_cols
        self.lags_cols = lags_cols
        self.model.set_cols(self.data_cols, self.lags_cols)
        self.model.reinit()

    def __call__(self, test, lags) -> np.ndarray:
        """
        Make predictions using the provided model
        """
        # Day complete and yesterday data is available for analysis now
        if lags is not None and self._day_data is not None:
            yesterday_data = self._day_end(lags)
            self.model.apply_yesterday_data(yesterday_data)

        # First item today
        time_id = test[0]['time_id'].item()
        if time_id == 0:
            self._day_start()

        test_data = self.df_symbols \
            .join(test, on='symbol_id', how='left') \
            .with_columns(
                time_sin=np.sin(2 * np.pi / self.num_timesteps * time_id),
                time_cos=np.cos(2 * np.pi / self.num_timesteps * time_id),
            ) \
            .select(self.data_cols) \
            .to_numpy()

        predictions = self.model.predict(test_data)

        # Accumulate data for future use
        self._day_accumulate(test, predictions)

        return predictions
