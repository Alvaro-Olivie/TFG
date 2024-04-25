import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('bonds.csv', low_memory=False)

def xgboost(X, y, params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("  starting to train model...")
    
    model = XGBRegressor(**params)

    
    model.fit(X_train, y_train)

    print("  model trained")
    
    y_pred = model.predict(X_test)

    print("  model predicted")
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    hit_ratio = (y_test * y_pred > 0).mean()

    print("  metrics calculated")
    
    coefs = pd.Series(model.feature_importances_)
    
    plt.figure()
    plt.title("Feature importances " + y.name + " returns")
    plt.barh(X.columns, coefs, color="r", align="center")
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("GradientBoosting/feature_importances_" + y.name + ".png", format="png")
    plt.close()

    y_pred = pd.Series(y_pred)
    y_pred.to_csv('GradientBoosting/y_pred_'+y.name+'.csv', index=False)
    
    return mse, r2, hit_ratio

def best_params(X, y):
    X_subset = X.sample(frac=0.1, random_state=42)
    y_subset = y[X_subset.index]
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)
    
    print("  starting to train model...")
    
    model = XGBRegressor(random_state=42, n_jobs=-1)
    
    params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.1, 0.01, 0.001],
        'booster': ['gbtree', 'gblinear', 'dart'],
        'max_depth': [3, 5, 7],
        'gamma': [0, 0.1, 0.2]
    }
    
    grid = GridSearchCV(model, params, n_jobs=6, cv=5)
    grid.fit(X_train, y_train)
    
    print("  best params: ", grid.best_params_)
    
    return grid.best_params_

if __name__ == "__main__":
    returns = ['R1M', 'R3M', 'R6M', 'R12M']
    results = pd.DataFrame(columns=['Target', 'MSE', 'R2', 'Hit Ratio'])

    for i in returns:
        print("Starting to process " + i)
        y = data[i]
        x = data.select_dtypes(include='number').drop(returns, axis=1)
        params = best_params(x, y)
        mse, r2, hit_ratio = xgboost(x, y, params)
        results = results._append({'Target': i, 'MSE': mse, 'R2': r2, 'Hit Ratio': hit_ratio}, ignore_index=True)
    results.to_csv('XGBoost/results.csv', index=False)
