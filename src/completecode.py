import pandas as pd
import yaml

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def save_processed(data: pd.DataFrame, out_path: str) -> None:
    data.to_csv(out_path, index=False)
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Drop missing
    df = df.dropna()

    # One-hot encode categorical
    cat_cols = df.select_dtypes(include="object").columns
    encoder = OneHotEncoder(sparse=False)
    encoded = encoder.fit_transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

    # Scale numeric
    numeric = df.select_dtypes(include=["int64", "float64"])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric)
    scaled_df = pd.DataFrame(scaled, columns=numeric.columns)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import joblib

def train_model(X, y, config: dict) -> dict:
    results = {}
    for name, params in config['model'].items():
        model = RandomForestRegressor() if name == 'random_forest' else GradientBoostingRegressor()
        grid = GridSearchCV(model, params, cv=5, scoring='r2')
        grid.fit(X, y)
        best = grid.best_estimator_
        results[name] = best
        joblib.dump(best, f"models/{name}_model.pkl")
    return results
import joblib
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

def evaluate_model(model_path: str, test_data_path: str):
    model = joblib.load(model_path)
    df = pd.read_csv(test_data_path)
    X = df.drop('Salary', axis=1)
    y = df['Salary']
    preds = model.predict(X)
    print(f"R2 Score: {r2_score(y, preds):.4f}")
    print(f"MSE: {mean_squared_error(y, preds):.2f}")
import pandas as pd
import numpy as np

# Reproducibility
np.random.seed(42)

# Number of samples
default_n = 6705

departments = ['Sales', 'Engineering', 'HR', 'Marketing']
education = ['Bachelors', 'Masters', 'PhD']
gender = ['Male', 'Female']

# Generate features
years_experience = np.random.normal(5, 2, default_n).clip(0)
deps = np.random.choice(departments, default_n)
edus = np.random.choice(education, default_n, p=[0.6,0.3,0.1])
gends = np.random.choice(gender, default_n, p=[0.7,0.3])

# Simulate salary
base = 30000 + years_experience * 2000
edu_bonus = np.array([5000 if e=='Masters' else 10000 if e=='PhD' else 0 for e in edus])
dep_bonus = np.array([3000 if d=='Engineering' else 0 for d in deps])
salary = base + edu_bonus + dep_bonus + np.random.normal(0, 2000, default_n)

# Build DF
df = pd.DataFrame({
    'YearsExperience': years_experience.round(1),
    'Department': deps,
    'Education': edus,
    'Gender': gends,
    'Salary': salary.round(2)
})

# Save CSV
import os
os.makedirs('data/raw', exist_ok=True)
df.to_csv('data/raw/salary_data.csv', index=False)
print(f"Generated data/raw/salary_data.csv with {len(df)} records")
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


