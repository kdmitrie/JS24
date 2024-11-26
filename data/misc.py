import polars as pl
from ..config.config import CFG

def get_df_smb(selector=None, symbol_id=None):
    df = pl.scan_parquet(CFG.path)

    if selector is not None:
        df = df[selector]

    if symbol_id is not None:
        df = df.filter(pl.col('symbol_id') == symbol_id)

    X = df.select(CFG.features).fill_null(strategy='forward').fill_null(0).collect().to_numpy()
    y = df.select(CFG.target).collect().to_numpy()
    weight = df.select(CFG.weight).collect().to_numpy()

    split = int(CFG.split_ratio * len(X))
    train_X, train_y, train_weight = X[:-split], y[:-split], weight[:-split]
    test_X, test_y, test_weight = X[-split:], y[-split:], weight[-split:]

    return train_X, test_X

def get_df_len():
    df = pl.scan_parquet(CFG.path)
    return df.select(pl.len()).collect().item()
