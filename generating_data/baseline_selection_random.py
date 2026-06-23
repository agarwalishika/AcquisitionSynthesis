import sys
sys.path.append('/home/ubuntu/AcquisitionSynthesis/')
from argparse import ArgumentParser
import os
from config import *

def filter(train_data: pd.DataFrame, size):
    return train_data.sample(n=size)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--data", type=str, default="numina")
    argparser.add_argument("--size", type=int, default=1000)
    args = argparser.parse_args()

    train_data = load_data(f"/home/ubuntu/AcquisitionSynthesis/data/{args.data}/all.parquet")

    filtered = filter(train_data, args.size)

    file_name = os.path.join(TRAINING_DATA_DIR, f"random_{args.data}_{str(args.size)}.parquet")
    
    def format(col):
        filtered[col] = filtered[col].apply(lambda x: x.replace("\n", ""))
    
    format('question')
    format('answer')
    format('reasoning')
    filtered.to_parquet(file_name)