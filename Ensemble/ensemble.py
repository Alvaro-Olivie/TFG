
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping


def nn_ensamble(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("  starting to train model...")

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['r2_score'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping])

    print("  model trained")

    y_pred = model.predict(X_test)
    y_pred = y_pred.reshape(-1)

    print("  model predicted")

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    hit_ratio = (y_test * y_pred > 0).mean()

    print("  metrics calculated")

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('MSE ' + y.name)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('Ensemble/loss_'+ y.name +'.png')
    plt.close()

    plt.plot(history.history['r2_score'])
    plt.plot(history.history['val_r2_score'])
    plt.title('R2 ' + y.name)
    plt.ylabel('R2')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('Ensemble/r2_'+ y.name +'.png')
    plt.close()

    return mse, r2, hit_ratio

def generate_df(targets):
    folders = ['GradientBoosting', 'NeuralNetwork', 'RandomForest', 'SVM', 'Regression']
    dfs = {}
    for i in targets:
        df = pd.DataFrame()
        for folder in folders:
            df['y_' + folder] = pd.read_csv(folder + '/y_pred_' + i + '.csv', usecols=[0])
        dfs[i] = df
    return dfs

def main():
    targets = ['R1M', 'R3M', 'R6M', 'R12M']
    dfs = generate_df(targets)

    results = pd.DataFrame(columns=['Target', 'MSE', 'R2', 'Hit Ratio'])
    data = pd.read_csv('bonds.csv', low_memory=False)
    x = data.select_dtypes(include='number').drop(targets, axis=1)

    for target, df in dfs.items():
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        print("Starting to process " + target)
        mse, r2, hit_ratio = nn_ensamble(df, y_test)
        results = results._append({'Target': target, 'MSE': mse, 'R2': r2, 'Hit Ratio': hit_ratio}, ignore_index=True)
    results.to_csv('Ensemble/results.csv', index=False)

if __name__ == '__main__':
    main()