import argparse
from src.data_processing import load_config, load_data, save_processed
from src.feature_engineering import preprocess
from src.model_training import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_data(cfg['data']['raw_path'])
    df_proc = preprocess(df)
    save_processed(df_proc, cfg['data']['processed_path'])
    X = df_proc.drop('Salary', axis=1)
    y = df_proc['Salary']
    train_model(X, y, cfg)

if __name__ == '__main__':
    main()
