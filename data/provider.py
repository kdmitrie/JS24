from dataclasses import dataclass
import numpy as np
import polars as pl
from typing import Callable, Optional


@dataclass
class JSDataProvider:
    path: str
    processor: Callable
    postprocessing: Optional[Callable] = None
    dates_to_process: int = 2

    def __post_init__(self) -> None:
        features = [f'feature_{n:02d}' for n in range(79)]
        responders = [f'responder_{n}' for n in range(9)]
        responders_lag_1 = [f'responder_{n}_lag_1' for n in range(9)]
        date_time_symbol = ['date_id', 'time_id', 'symbol_id']
        time_sin_cos = ['time_sin', 'time_cos']

        self.data_cols = ['row_id', 'weight'] + date_time_symbol + features
        self.lags_cols = date_time_symbol + responders
        self.lags_renamed_cols = date_time_symbol + responders_lag_1
        self.validation_cols = ['row_id', 'weight', 'responder_6']

        self.ext_data_cols = ['row_id', 'weight'] + date_time_symbol + time_sin_cos + features
        self.ext_lags_renamed_cols = ['row_id', 'weight',
                                      'responder_6'] + date_time_symbol + time_sin_cos + features + responders_lag_1
        self.scores = []
        self.processor.set_cols(self.ext_data_cols, self.ext_lags_renamed_cols)

    @staticmethod
    def make_asserts(predictions: pl.DataFrame, X: pl.DataFrame) -> None:
        # The predict function must return a DataFrame
        assert isinstance(predictions, pl.DataFrame)

        # with columns 'row_id', 'responer_6'
        assert predictions.columns == ['row_id', 'responder_6']

        # and as many rows as the test data.
        assert len(predictions) == len(X)

    @staticmethod
    def score(y_true: pl.DataFrame, y_pred: pl.DataFrame) -> float:
        gt = y_true['responder_6']
        w = y_true['weight']
        pred = y_pred['responder_6']
        return 1 - ((pred - gt).pow(2) * w).sum() / (gt.pow(2) * w).sum()

    def execute(self) -> None:
        # 1. Select all data
        df = pl.scan_parquet(self.path)
        all_dates = sorted(df.select(pl.col("date_id").unique()).collect().get_column("date_id"))

        if self.dates_to_process:
            all_dates = all_dates[:self.dates_to_process]

        print(f'1. Loaded {len(all_dates)} dates')

        # Initial values
        lags = None
        start_row_id = 0
        self.scores = []

        # Date loop
        for date_id in all_dates:
            print(f'2. Processing date={date_id}')
            df_day = df.filter(pl.col('date_id') == date_id).collect()

            # Generate `row_id` column
            date_len = len(df_day)
            row_id = pl.Series('row_id', start_row_id + np.arange(date_len))
            df_day.insert_column(0, row_id)
            start_row_id += date_len

            # Generate batches based on `time_id`
            batches = df_day.group_by('time_id', maintain_order=True)

            y_true = df_day.select(self.validation_cols)
            y_pred = []

            # Time loop
            for (time_id,), batch in batches:
                X = batch.select(self.data_cols)
                X_lags = lags if time_id == 0 else None

                # Get the prediction
                y = self.processor(X, X_lags)

                # Asserts everything seems ok
                self.make_asserts(y, X)

                y_pred.append(y)

            y_pred = pl.concat(y_pred)
            score = self.score(y_true, y_pred)
            self.scores.append(score)
            print(f'\tscore = {score:.4f}')

            lags = df_day.select(self.lags_cols).rename(dict(zip(self.lags_cols, self.lags_renamed_cols)))

        if self.postprocessing is not None:
            self.postprocessing(self.scores)
