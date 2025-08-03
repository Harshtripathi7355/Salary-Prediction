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
