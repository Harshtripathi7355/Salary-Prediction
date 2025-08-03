import pandas as pd
import yaml

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def save_processed(data: pd.DataFrame, out_path: str) -> None:
    data.to_csv(out_path, index=False)
