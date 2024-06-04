import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import joblib
import matplotlib.pyplot as plt

def svm_model(X, y, params):
    X_scaled = preprocessing.scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("  starting to train model...")

    model = svm.SVR(**params)
    
    model.fit(X_train, y_train)
    
    print("  model trained")
    
    y_pred = model.predict(X_test)

    print("  model predicted")
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    hit_ratio = (y_test * y_pred > 0).mean()

    print("  metrics calculated")

    plt.scatter(y_test, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True Values vs Predictions for ' + y.name)
    plt.savefig('SVM/scatter_'+ y.name +'.png')
    plt.close()

    y_pred = pd.Series(y_pred)
    y_pred.to_csv('SVM/y_pred_'+y.name+'.csv', index=False)

    # Store the trained model
    joblib.dump(model, 'SVM/model'+y.name+'.pkl')
    
    return mse, r2, hit_ratio

def best_params(X, y):
    X_subset = X.sample(frac=0.1, random_state=42)
    y_subset = y[X_subset.index]
    X_scaled = preprocessing.scale(X_subset)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_subset, test_size=0.2, random_state=42)
    
    print("  starting to train model...")
    
    model = svm.SVR()
    
    params = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.05, 0.1],
    }
    
    grid= GridSearchCV(model, params, n_jobs=6)
    grid.fit(X_train, y_train)
    
    print("  best params: ", grid.best_params_)
    
    return grid.best_params_
    
def main():
    data = pd.read_csv('test_bonds.csv', low_memory=False)
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

if __name__ == "__main__":
    main()