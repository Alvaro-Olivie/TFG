import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('bonds.csv', low_memory=False)

def svm_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("  starting to train model...")

    model = svm.SVR()
    
    params = {
        'C': [0.1, 1, 10, 100, 1000],
        'epsilon': [0.1, 0.2, 0.3, 0.4, 0.5],
        'gamma': ['scale', 'auto']
    }
    
    grid_search = GridSearchCV(model, params, n_jobs=6)
    
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    
    model.fit(X_train, y_train)
    print("  model trained")
    
    y_pred = model.predict(X_test)

    print("  model predicted")
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    hit_ratio = (y_test * y_pred > 0).mean()

    print("  metrics calculated")
    
    return mse, r2, hit_ratio
    

if __name__ == "__main__":
    returns = ['R1M', 'R3M', 'R6M', 'R12M']
    results = pd.DataFrame(columns=['Target', 'MSE', 'R2', 'Hit Ratio'])

    for i in returns:
        print("Starting to process " + i)
        y = data[i]
        x = data.select_dtypes(include='number').drop(returns, axis=1)
        mse, r2, hit_ratio = svm_model(x, y)
        results = results._append({'Target': i, 'MSE': mse, 'R2': r2, 'Hit Ratio': hit_ratio}, ignore_index=True)
    results.to_csv('SVM/results.csv', index=False)