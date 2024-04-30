import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

targets = ['R1M', 'R3M', 'R6M', 'R12M']
folders = ['GradientBoosting', 'NeuralNetwork', 'RandomForest', 'SVM', 'Regression']
data = pd.read_csv('bonds.csv', low_memory=False)
x = data.select_dtypes(include='number').drop(targets, axis=1)
for i in targets:
        y = data[i]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        for folder in folders:
            y_pred = pd.read_csv(folder + '/y_pred_' + i + '.csv', usecols=[0])
            plt.scatter(y_test, y_pred)
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title('True Values vs Predictions for ' + y.name)
            plt.savefig(folder + '/scatter_'+ y.name +'.png')
            plt.close()

