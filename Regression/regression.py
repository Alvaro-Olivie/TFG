import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('bonds.csv', low_memory=False)

def linear_regression(X, y):
    X_scaled = preprocessing.scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    hit_ratio = (y_test * y_pred > 0).mean()

    coefs = pd.Series(abs(model.coef_))

    plt.figure()
    plt.title("Feature importances " + y.name + " returns")
    plt.barh(X.columns, coefs, color="r", align="center")
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.savefig("Regression/feature_importances_" + y.name + ".png", format="png")
    plt.close()

    return mse, r2, hit_ratio


def ridge_regression(X, y):
    X_scaled = preprocessing.scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = Ridge()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    hit_ratio = (y_test * y_pred > 0).mean()

    coefs = pd.DataFrame(model.coef_)

    print(coefs)

    plt.figure()
    plt.title("Coefficient values vs Penalization for " + y.name + " returns")
    plt.plot(coefs)
    plt.xlabel("Penalization")
    plt.ylabel("Coefficient values")
    plt.legend(X.columns)
    plt.savefig("Regression/coefficient_values_" + y.name + ".png", format="png")
    plt.close()

    return mse, r2, hit_ratio


if __name__ == '__main__':
    returns = ['R1M', 'R3M', 'R6M', 'R12M']
    results = pd.DataFrame(columns=['Model', 'Y', 'MSE', 'R2', 'Hit Ratio'])

    for i in returns:
        y = data[i]
        x = data.select_dtypes(include='number').drop(returns, axis=1)
        
        mse, r2, hit_ratio = linear_regression(x, y)
        results = results._append({'Model': 'Linear Regression', 'Y': i, 'MSE': mse, 'R2': r2, 'Hit Ratio': hit_ratio}, ignore_index=True)
        
        mse_ridge, r2_ridge, hit_ratios_ridge = ridge_regression(x, y)
        results = results._append({'Model': 'Ridge Regression', 'Y': i, 'MSE': mse_ridge, 'R2': r2_ridge, 'Hit Ratio': hit_ratios_ridge}, ignore_index=True)

    print(results)



        
        