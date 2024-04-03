
import pandas as pd
import matplotlib.pyplot as plt

def plot_column(df, column):
    for i in column:
        Fig, ax = plt.subplots(figsize=(10, 4))

        df.groupby('Date')[i].mean().plot(ax=ax)
        ax.set_ylabel(i)
        ax.set_xlabel('Date')
        ax.set_title(i + ' Time Series')
        plt.savefig('DataPreprocessing/' + i + '_Time.png')

if __name__ == '__main__':
    df = pd.read_csv('bonds.csv', low_memory=False)
    plot_column(df, ['Price', 'R1M','R3M','R6M','R12M'])

