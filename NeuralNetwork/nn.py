import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
import matplotlib.pyplot as plt

def nn(X, y):
    X_scaled = preprocessing.scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("  starting to train model...")
    
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    

    model.compile(optimizer='RMSprop', 
              loss='mse',
              metrics=['r2_score'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, epochs=100, batch_size=256, validation_split=0.2, verbose=1, callbacks=[early_stopping])

    print("  model trained")
    
    y_pred = model.predict(X_test)

    # y_pred has shape (n, 1), we need to reshape it to (n,)
    y_pred = y_pred.reshape(-1)

    print("  model predicted")
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    hit_ratio = (y_test * y_pred > 0).mean()

    print("  metrics calculated")

    # plot the loss and the val loss over time
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('MSE ' + y.name)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('NeuralNetwork/loss_'+ y.name +'.png')
    plt.close()

    plt.plot(history.history['r2_score'])
    plt.plot(history.history['val_r2_score'])
    plt.title('R2 ' + y.name)
    plt.ylabel('R2')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('NeuralNetwork/r2_'+ y.name +'.png')
    plt.close()

    plt.scatter(y_test, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True Values vs Predictions for ' + y.name)
    plt.savefig('NeuralNetwork/scatter_'+ y.name +'.png')
    plt.close()

    y_pred = pd.Series(y_pred)
    y_pred.to_csv('NeuralNetwork/y_pred_'+y.name+'.csv', index=False)

    model.save('NeuralNetwork/model'+y.name+'.h5')
    
    return mse, r2, hit_ratio

def main():
    returns = ['R1M', 'R3M', 'R6M', 'R12M']
    results = pd.DataFrame(columns=['Target', 'MSE', 'R2', 'Hit Ratio'])

    data = pd.read_csv('test_bonds.csv', low_memory=False)
    
    for i in returns:
        print("Starting to process " + i)
        y = data[i]
        x = data.select_dtypes(include='number').drop(returns, axis=1)
        mse, r2, hit_ratio = nn(x, y)
        results = results._append({'Target': i, 'MSE': mse, 'R2': r2, 'Hit Ratio': hit_ratio}, ignore_index=True)
    results.to_csv('NeuralNetwork/results.csv', index=False)

if __name__ == "__main__":
    main()

