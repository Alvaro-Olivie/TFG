import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score


def nn(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("  starting to train model...")
    
    model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
    
    model.fit(X_train, y_train)

    print("  model trained")
    
    y_pred = model.predict(X_test)

    print("  model predicted")
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    hit_ratio = (y_test * y_pred > 0).mean()

    print("  metrics calculated")
    
    coefs = pd.Series(model.coefs_[0].mean(axis=1))
    
    plt.figure()
    plt.title("Feature importances " + y.name + " returns")
    plt.barh(X.columns, coefs, color="r", align="center")
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.savefig("NeuralNetwork/feature_importances_" + y.name + ".png", format="png")
    plt.close()
    
    return mse, r2, hit_ratio

if __name__ == "__main__":
    returns = ['R1M', 'R3M', 'R6M', 'R12M']
    results = pd.DataFrame(columns=['Target', 'MSE', 'R2', 'Hit Ratio'])

    data = pd.read_csv('bonds.csv', low_memory=False)
    
    for i in returns:
        print("Starting to process " + i)
        y = data[i]
        x = data.select_dtypes(include='number').drop(returns, axis=1)
        mse, r2, hit_ratio = nn(x, y)
        results = results.append({'Target': i, 'MSE': mse, 'R2': r2, 'Hit Ratio': hit_ratio}, ignore_index=True)
    results.to_csv('NeuralNetwork/results.csv', index=False)