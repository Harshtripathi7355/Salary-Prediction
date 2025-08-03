import argparse
from src.evaluation import evaluate_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--test-data', default='data/processed/processed.csv')
    args = parser.parse_args()

    evaluate_model(args.model_path, args.test_data)

if __name__ == '__main__':
    main()
