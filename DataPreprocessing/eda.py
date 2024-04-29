
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_seasonal(df):
    data_avg = df.groupby('Date')['Price'].mean().reset_index()
    data_avg = data_avg.set_index('Date')

    result = seasonal_decompose(data_avg['Price'], model='additive', period=12)

    # now plot the result
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
    result.observed.plot(ax=ax1)
    ax1.set_ylabel('Observed')
    result.trend.plot(ax=ax2)
    ax2.set_ylabel('Trend')
    result.seasonal.plot(ax=ax3)
    ax3.set_ylabel('Seasonal')
    result.resid.plot(ax=ax4)
    ax4.set_ylabel('Residual')
    plt.savefig('DataPreprocessing/seasonal_decompose.png')

def plot_column(df, column):
    for i in column:
        Fig, ax = plt.subplots(figsize=(10, 4))

        df.groupby('Date')[i].mean().plot(ax=ax)
        ax.set_ylabel(i)
        ax.set_xlabel('Date')
        ax.set_title(i + ' Time Series')
        plt.savefig('DataPreprocessing/' + i + '_Time.png')

def plot_distribution(df, column):
    df.drop_duplicates(subset='ISIN', keep='first', inplace=True)
    for i in column:
        Fig, ax = plt.subplots(figsize=(10, 4))
        df[i].hist(ax=ax)
        ax.set_ylabel('Frequency')
        ax.set_xlabel(i)
        ax.set_title(i + ' Distribution')
        plt.tight_layout()
        plt.savefig('DataPreprocessing/' + i + '_Distribution.png')



if __name__ == '__main__':
    df = pd.read_csv('bonds.csv', low_memory=False)
    plot_column(df, ['Price', 'R1M','R3M','R6M','R12M'])
    plot_seasonal(df)
    plot_distribution(df, ['Cpn', 'Maturity', 'Index Rating', 'R1M', 'R3M', 'R6M', 'R12M'])


