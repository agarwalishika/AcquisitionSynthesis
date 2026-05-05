import pandas as pd

datasets = ['alpaca', 'openr1math']

for split in ['train', 'valid', 'test']:
    ds = []
    for dataset in datasets:
        temp = pd.read_parquet(f"{dataset}/{split}.parquet", engine='pyarrow')
        ds.append(temp[:len(temp)//2])
    
    df = pd.concat(ds)
    df.to_parquet(f'combined_alpaca_openr1math/{split}.parquet')