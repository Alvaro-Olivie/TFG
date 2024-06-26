import pandas as pd
import joblib
from sklearn import preprocessing
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def regression(data):
    print('Regression')
    returns = ['R1M', 'R3M', 'R6M', 'R12M']
    reg = joblib.load('Regression/modelR1M.pkl')
    
    results = pd.DataFrame()

    for month in data['Date'].unique():
        month_data = data[data['Date'] == month]
        X = month_data.select_dtypes(include='number').drop(returns, axis=1)
        X_scaled = preprocessing.scale(X)
        y_pred = reg.predict(X_scaled)
        top_predictions_index = y_pred.argsort()[-int(len(y_pred) * 0.15):]

        top_predictions_data = month_data.iloc[top_predictions_index]
        results = pd.concat([results, top_predictions_data], ignore_index=True)
    return results

def svm(data):
    print('SVM')
    returns = ['R1M', 'R3M', 'R6M', 'R12M']
    model = joblib.load('SVM/modelR1M.pkl')
    
    results = pd.DataFrame()

    for month in data['Date'].unique():
        month_data = data[data['Date'] == month]
        X = month_data.select_dtypes(include='number').drop(returns, axis=1)
        X_scaled = preprocessing.scale(X)
        y_pred = model.predict(X_scaled)
        top_predictions_index = y_pred.argsort()[-int(len(y_pred) * 0.15):]
        top_predictions_data = month_data.iloc[top_predictions_index]
        results = pd.concat([results, top_predictions_data], ignore_index=True)
    return results
    
def randomForest(data):
    print('RF')
    returns = ['R1M', 'R3M', 'R6M', 'R12M']
    svm_model = joblib.load('RandomForest/modelR1M.pkl')
    
    results = pd.DataFrame()

    for month in data['Date'].unique():
        month_data = data[data['Date'] == month]
        X = month_data.select_dtypes(include='number').drop(returns, axis=1)
        y_pred = svm_model.predict(X)
        top_predictions_index = y_pred.argsort()[-int(len(y_pred) * 0.15):]
        top_predictions_data = month_data.iloc[top_predictions_index]
        results = pd.concat([results, top_predictions_data], ignore_index=True)
    return results

def xgboost(data):
    print('GB')
    returns = ['R1M', 'R3M', 'R6M', 'R12M']
    model = joblib.load('GradientBoosting/modelR1M.pkl')
    
    results = pd.DataFrame()

    for month in data['Date'].unique():
        month_data = data[data['Date'] == month]
        X = month_data.select_dtypes(include='number').drop(returns, axis=1)
        y_pred = model.predict(X)
        top_predictions_index = y_pred.argsort()[-int(len(y_pred) * 0.15):]
        top_predictions_data = month_data.iloc[top_predictions_index]
        results = pd.concat([results, top_predictions_data], ignore_index=True)
    return results

def neuralNetwork(data):
    print('Neural Network')
    returns = ['R1M', 'R3M', 'R6M', 'R12M']
    custom_objects = {
        'mse': tf.keras.metrics.MeanSquaredError(name='mse')
    }
    model = tf.keras.models.load_model('NeuralNetwork/modelR1M.h5', custom_objects=custom_objects)
    
    results = pd.DataFrame()

    for month in data['Date'].unique():
        month_data = data[data['Date'] == month]
        X = month_data.select_dtypes(include='number').drop(returns, axis=1)
        X_scaled = preprocessing.scale(X)
        y_pred = model.predict(X_scaled)
        y_pred = y_pred.flatten()  # Ensure y_pred is 1-dimensional
        top_predictions_index = np.argsort(y_pred)[-int(len(y_pred) * 0.15):]
        top_predictions_data = month_data.iloc[top_predictions_index]
        results = pd.concat([results, top_predictions_data], ignore_index=True)
    return results

def plot_cumulative_returns(df, label):
    cumulative_return = (1 + df['R1M']).cumprod() * 100
    cumulative_return.plot(label=label)
    return cumulative_return

def main():
    data = pd.read_csv('test_bonds.csv')
    
    reg = regression(data)
    svm_model = svm(data)
    rf = randomForest(data)
    gb = xgboost(data)
    nn = neuralNetwork(data)

    # Convert date columns to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    reg['Date'] = pd.to_datetime(reg['Date'])
    svm_model['Date'] = pd.to_datetime(svm_model['Date'])
    rf['Date'] = pd.to_datetime(rf['Date'])
    gb['Date'] = pd.to_datetime(gb['Date'])
    nn['Date'] = pd.to_datetime(nn['Date'])

    # Calculate weighted market return
    data['Weighted_R1M'] = data['R1M'] * data['Weight']
    market_returns = data.groupby('Date').apply(lambda x: (x['Weighted_R1M'].sum() / x['Weight'].sum())).reset_index(name='R1M')

    # Plot average R1M by Date
    plt.figure(figsize=(10, 6))
    market_returns.set_index('Date')['R1M'].plot(label='Market')
    reg.groupby('Date')['R1M'].mean().plot(label='LR')
    svm_model.groupby('Date')['R1M'].mean().plot(label='SVR')
    rf.groupby('Date')['R1M'].mean().plot(label='RF')
    gb.groupby('Date')['R1M'].mean().plot(label='GB')
    nn.groupby('Date')['R1M'].mean().plot(label='NN')
    plt.xlabel('Date')
    plt.ylabel('R1M')
    plt.title('Average R1M by Date')
    plt.legend()
    plt.savefig("Portfolio/average_R1M.png", format="png")
    plt.close()

    # Plot cumulative returns assuming a $100 investment
    plt.figure(figsize=(10, 6))
    market_cumulative = plot_cumulative_returns(market_returns, 'Market')
    reg_cumulative = plot_cumulative_returns(reg.groupby('Date')['R1M'].mean().reset_index(), 'LR')
    svm_cumulative = plot_cumulative_returns(svm_model.groupby('Date')['R1M'].mean().reset_index(), 'SVR')
    rf_cumulative = plot_cumulative_returns(rf.groupby('Date')['R1M'].mean().reset_index(), 'RF')
    gb_cumulative = plot_cumulative_returns(gb.groupby('Date')['R1M'].mean().reset_index(), 'GB')
    nn_cumulative = plot_cumulative_returns(nn.groupby('Date')['R1M'].mean().reset_index(), 'NN')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return ($)')
    plt.title('Cumulative Returns of Investment')
    plt.legend()
    plt.savefig("Portfolio/cumulative_returns_R1M.png", format="png")
    plt.close()

    # Print last values of cumulative returns
    print(f"Market final cumulative return: ${market_cumulative.iloc[-1]:.2f}")
    print(f"Linear Regression final cumulative return: ${reg_cumulative.iloc[-1]:.2f}")
    print(f"SVM final cumulative return: ${svm_cumulative.iloc[-1]:.2f}")
    print(f"Random Forest final cumulative return: ${rf_cumulative.iloc[-1]:.2f}")
    print(f"Gradient Boosting final cumulative return: ${gb_cumulative.iloc[-1]:.2f}")
    print(f"Neural Network final cumulative return: ${nn_cumulative.iloc[-1]:.2f}")

if __name__ == '__main__':
    main()
