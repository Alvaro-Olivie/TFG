import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def linear_regression(X, y):
    X_scaled = preprocessing.scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("  starting to train model...")

    model = LinearRegression()

    model.fit(X_train, y_train)

    print("  model trained")

    y_pred = model.predict(X_test)

    print("  model predicted")

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    hit_ratio = (y_test * y_pred > 0).mean()

    print("  metrics calculated")

    coefs = pd.Series(abs(model.coef_))

    plt.figure(figsize=(12, 8))
    plt.title("Feature importances " + y.name + " returns")
    plt.barh(X.columns, coefs, align="center")
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.tight_layout() 
    plt.savefig("Regression/feature_importances_" + y.name + ".png", format="png")
    plt.close()

    plt.scatter(y_test, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True Values vs Predictions for ' + y.name)
    plt.savefig('Regression/scatter_'+ y.name +'.png')
    plt.close()

    y_pred = pd.Series(y_pred)
    y_pred.to_csv('Regression/y_pred_'+y.name+'.csv', index=False)

    # Store the trained model
    joblib.dump(model, 'Regression/model'+y.name+'.pkl')

    return mse, r2, hit_ratio


def lasso_regression(X, y_penalized): 
    X_penalized = preprocessing.scale(X.values)
    n_alphas = 50 # declare the number of alphas for ridge
    alphas = np.logspace(-4,-2,n_alphas) # here alpha is used for lambda in scikit-learn
    lasso_res = {} # declaring the dict that will receive the model's 

    print("Starting to process Lasso Regression for " + y_penalized.name) 

    for alpha in alphas: # looping through the different alphas/lambdas values
        lasso = Lasso(alpha=alpha) # model
        lasso.fit(X_penalized,y_penalized) 
        lasso_res[alpha] = lasso.coef_ # extract LASSO coefs

    df_lasso_res = pd.DataFrame.from_dict(lasso_res).T # transpose the dataframe for plotting
    df_lasso_res.columns = X.columns # adding the names of the factors
    predictors = (df_lasso_res.abs().sum() > 0.05) # selecting the most relevant
    df_lasso_res.loc[:,predictors].plot(xlabel='Lambda',ylabel='Beta',figsize=(12,8)); # Plot!

    plt.figure()
    plt.title("Coefficient values vs Lasso Penalization for " + y_penalized.name + " returns")
    df_lasso_res.loc[:,predictors].plot(xlabel='Lambda',ylabel='Beta',figsize=(13,8)) # Plot!
    plt.xlabel("Penalization")
    plt.ylabel("Coefficient values")    
    plt.legend(X.columns)
    plt.savefig("Regression/lasso_values_" + y_penalized.name + ".png", format="png")
    plt.close()

    print("  Lasso Regression for " + y_penalized.name + " returns processed")
    

def ridge_regression(X, y_penalized):
    X_penalized = preprocessing.scale(X.values)
    n_alphas = 50 # declare the number of alphas for ridge
    alphas = np.logspace(-2, 5, n_alphas) # transforming into log for Aspect ratio 
    ridge_res = {} # declaring the dict that will receive the model's result 

    print("Starting to process Ridge Regression for " + y_penalized.name)

    for alpha in alphas: # looping through the different alphas/lambdas values
        ridge = Ridge(alpha=alpha) # model
        ridge.fit(X_penalized,y_penalized) # fit the model
        ridge_res[alpha] = ridge.coef_ # extract RIDGE coefs

    df_ridge_res = pd.DataFrame.from_dict(ridge_res).T # transpose the dataframe for plotting
    df_ridge_res.columns = X.columns # adding the names of the factors
    predictors = (df_ridge_res.abs().sum() > 0.05) # selecting the most relevant

    plt.figure()
    plt.title("Coefficient values vs Ridge Penalization for " + y_penalized.name + " returns")
    df_ridge_res.loc[:,predictors].plot(xlabel='Lambda',ylabel='Beta',figsize=(13,8)) # Plot!
    plt.xlabel("Penalization")
    plt.ylabel("Coefficient values")    
    plt.legend(X.columns)
    plt.savefig("Regression/ridge_values_" + y_penalized.name + ".png", format="png")
    plt.close()

    print("  Ridge Regression for " + y_penalized.name + " returns processed")

def main():
    data = pd.read_csv('test_bonds.csv', low_memory=False)
    returns = ['R1M', 'R3M', 'R6M', 'R12M']
    results = pd.DataFrame(columns=['Target', 'MSE', 'R2', 'Hit Ratio'])

    for i in returns:
        print("Starting to process " + i)
        y = data[i]
        x = data.select_dtypes(include='number').drop(returns, axis=1)
        
        mse, r2, hit_ratio = linear_regression(x, y)
        results = results._append({'Target': i, 'MSE': mse, 'R2': r2, 'Hit Ratio': hit_ratio}, ignore_index=True)
        
        ridge_regression(x, y)
        lasso_regression(x, y)

    results.to_csv('Regression/results.csv', index=False)

if __name__ == '__main__':
    main()


        
        