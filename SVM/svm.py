import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('bonds.csv', low_memory=False)

def svm_model(X, y, params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("  starting to train model...")
    
    model = svm.SVR(**params, n_jobs=6, random_state=42)
    
    model.fit(X_train, y_train)

    print("  model trained")
    
    y_pred = model.predict(X_test)

    print("  model predicted")
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    hit_ratio = (y_test * y_pred > 0).mean()

    print("  metrics calculated")
    
    coefs = pd.Series(model.coef_)
    
    plt.figure()
    plt.title("Feature importances " + y.name + " returns")
    plt.barh(X.columns, coefs, color="r", align="center")
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.savefig("SVM/feature_importances_" + y.name + ".png", format="png")
    plt.close()
    
    return mse, r2, hit_ratio

def best_params(X, y):
    X_subset = X.sample(frac=0.1, random_state=42)
    y_subset = y[X_subset.index]
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)
    
    print("  starting to train model...")
    
    model = svm.SVR()
    
    params = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4, 5],
        'C': [0.1, 1, 10, 100, 1000],
        'epsilon': [0.1, 0.2, 0.3, 0.4, 0.5],
        'gamma': ['scale', 'auto']
    }
    
    grid_search = GridSearchCV(model, params, n_jobs=6, verbose=2)
    
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("  best model selected:", best_model)

    return best_model

    

if __name__ == "__main__":
    returns = ['R1M', 'R3M', 'R6M', 'R12M']
    results = pd.DataFrame(columns=['Target', 'MSE', 'R2', 'Hit Ratio'])

    for i in returns:
        print("Starting to process " + i)
        y = data[i]
        x = data.select_dtypes(include='number').drop(returns, axis=1)
        params = best_params(x, y)
        mse, r2, hit_ratio = svm_model(x, y, params)
        results = results._append({'Target': i, 'MSE': mse, 'R2': r2, 'Hit Ratio': hit_ratio}, ignore_index=True)
    results.to_csv('SVM/results.csv', index=False)