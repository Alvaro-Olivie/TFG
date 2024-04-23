import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('bonds.csv', low_memory=False)

def random_forest(X, y, params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("  starting to train model...")
    
    model = RandomForestRegressor(random_state=42, n_jobs=6, **params)

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
    plt.savefig("RandomForest/feature_importances_" + y.name + ".png", format="png")
    plt.close()
    
    return mse, r2, hit_ratio

def decision_tree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("  starting to train model...")

    model = DecisionTreeRegressor(random_state=42, max_depth=3)
    
    model.fit(X_train, y_train)

    print("  model trained")
    
    y_pred = model.predict(X_test)

    print("  model predicted")
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    hit_ratio = (y_test * y_pred > 0).mean()

    print("  metrics calculated")

    plt.figure(figsize=(17, 10))
    tree.plot_tree(model, feature_names=X.columns, filled=True, rounded=True, fontsize=11, class_names=True, proportion=True)
    plt.savefig("RandomForest/decision_tree_" + y.name + ".png", format="png", dpi=300, bbox_inches='tight')
    plt.close()

    print("  tree saved")

    return mse, r2, hit_ratio

def best_params(X, y):
    X_subset = X.sample(frac=0.1, random_state=42)
    y_subset = y[X_subset.index]
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)

    print("  starting to train model...")
    
    params_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 2, 3, 4]
    }

    model = RandomForestRegressor(random_state=42, n_jobs=6)

    grid_search = GridSearchCV(estimator=model, param_grid=params_grid, cv=3, scoring='neg_mean_squared_error')
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
        mse, r2, hit_ratio = random_forest(x, y, params)
        results = results._append({'Target': i, 'MSE': mse, 'R2': r2, 'Hit Ratio': hit_ratio}, ignore_index=True)
    results.to_csv('RandomForest/results_rf.csv', index=False)


    results = pd.DataFrame(columns=['Target', 'MSE', 'R2', 'Hit Ratio'])

    for i in returns:
        print("Starting to process " + i)
        y = data[i]
        x = data.select_dtypes(include='number').drop(returns, axis=1)
        mse, r2, hit_ratio = decision_tree(x, y)
        results = results._append({'Target': i, 'MSE': mse, 'R2': r2, 'Hit Ratio': hit_ratio}, ignore_index=True)
    results.to_csv('RandomForest/results_dt.csv', index=False)

        