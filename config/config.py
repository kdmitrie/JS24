class CFG:
    features = [f'feature_{n:02d}' for n in range(79)]
    responders = [f'responder_{n}' for n in range(9) if n !=6]
    columns = features + responders
    target = 'responder_6'
    weight = 'weight'
    path = '/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet'
    num_symbols = 50
    split_ratio = 0.1
