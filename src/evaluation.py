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
