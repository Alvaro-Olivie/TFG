import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_excel_sheets(path):
    # Open the excel file
    xl = pd.ExcelFile(path)
    # Create an empty list to store the dataframes
    dfs = []
    # Loop through each sheet in the excel file
    for sheet_name in xl.sheet_names:
        # Read the sheet into a dataframe
        df = pd.read_excel(xl, sheet_name)
        # Add a column with the sheet name
        # First parse the sheet name so it´s a date. It comes in the format: "short month day year"
        if sheet_name[0:4] == "Aprl":
            sheet_name = "April"+sheet_name[4:]
        df['Date'] = pd.to_datetime(sheet_name, format='mixed')
        # Append the dataframe to the list
        dfs.append(df)
    # Concatenate all the dataframes into a single dataframe
    result = pd.concat(dfs, ignore_index=True)
    return result

def clean_data(data):
    data['Payment Rank'] = data['Payment rank'].fillna(data['payment rank'])
    data = data.drop(columns=['Payment rank', 'payment rank'])
    data['ISIN'] = data['ISIN'].fillna(data['Unnamed: 0'])
    data = data.drop(columns=['Unnamed: 0'])
    data['Cpn'] = data['Cpn'].fillna(data['Coupon'])
    data = data.drop(columns=['Coupon'])
    data = data.rename(columns={'Maturity.1': 'Maturity Date'})
    data = data.drop(columns=['Description', 'Ccy', 'Issuer', 'Maturity Date'])
    data.rename(columns={'Yield to Maturity': 'YTM'}, inplace=True)
    return data

def categorical_to_numeric(data):
    data = data[data['Payment Rank'].str.contains('1st lien', na=False)]
    data = data.drop(columns=['Payment Rank'])
    data['Index Rating (String)'] = data['Index Rating (String)'].replace(['A2', 'A3', 'BAA3', 'BAA1', 'BAA2', 'AA3', 'A1', 'AA2',  'AA1', 'AAA'], 'Investment Grade')
    data['Index Rating (String)'] = data['Index Rating (String)'].replace(['BA3', 'B3','BA1', 'B1', 'BA2', 'CAA1', 'CAA2', 'CAA3', 'NR', 'CA', 'D', 'C', 'B2'], 'High Yield')
    data['Index Rating (String)'] = data['Index Rating (String)'].map({'Investment Grade': 1, 'High Yield': 0})
    data = data.rename(columns={'Index Rating (String)': 'Index Rating'})
    data['BCLASS 2'] = data['BCLASS 2'].map({'FINANCIAL_INSTITUTIONS': 1, 'INDUSTRIAL': 2, 'UTILITY': 3})
    return data

def calculate_returns(data):    
    data = data.sort_values(['ISIN', 'Date'])
    data['R1M'] = data.groupby('ISIN')['Price'].pct_change()
    data['R1M'] = data.groupby('ISIN')['R1M'].shift(-1)
    data['R3M'] = data.groupby('ISIN')['Price'].pct_change(3)
    data['R3M'] = data.groupby('ISIN')['R3M'].shift(-3)
    data['R6M'] = data.groupby('ISIN')['Price'].pct_change(6)
    data['R3M'] = data.groupby('ISIN')['R3M'].shift(-6)
    data['R12M'] = data.groupby('ISIN')['Price'].pct_change(12)
    data['R3M'] = data.groupby('ISIN')['R3M'].shift(-12)
    return data.replace([np.inf, -np.inf], np.nan).dropna()

def plot_categorical_distribution(df, column):
    df.drop_duplicates(subset='ISIN', keep='first', inplace=True)
    for i in column:
        Fig, ax = plt.subplots(figsize=(10, 4))
        df[i].hist(ax=ax)
        ax.set_ylabel('Frequency')
        ax.set_xlabel(i)
        ax.set_xticks(range(len(df[i].unique())))
        ax.set_xticklabels(sorted(df[i].unique()), rotation=60)
        ax.set_title(i + ' Distribution')
        plt.tight_layout()
        plt.savefig('DataPreprocessing/' + i + '_Distribution.png')

def feature_correlation(df):
    correlation = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(correlation, cmap='coolwarm')
    fig.colorbar(cax)
    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
    plt.yticks(range(len(correlation.columns)), correlation.columns)
    plt.savefig('DataPreprocessing/correlation_matrix.png')

def main():
    dfs = []
    xc = ["Data\LUACTRUU Index 2018-2020.xlsx", 
          "Data\LUACTRUU Index 2023-2021.xlsx", 
          "Data\LUACTRUU Index Data (2015-2013).xlsx", 
          "Data\LUUACTRUU Index 2017-2016.xlsx", 
          "Data\LF98TRUU Index 2013.xlsx", 
          "Data\LF98TRUU Index (2014-2023).xlsb"]
    
    for i in xc:
        dfs.append(read_excel_sheets(i))
        print(i)
    data = pd.concat(dfs, ignore_index=True)
    data = clean_data(data)
    #plot_categorical_distribution(data, ['Index Rating (String)', 'BCLASS 2', 'Payment Rank'])
    #data = categorical_to_numeric(data)
    #data = calculate_returns(data)
    #feature_correlation(data.drop(columns=['ISIN', 'Date']))

    #high correlation with YTW, Par Val and OAD
    #data = data.drop(columns=['YTW', 'Par Val', 'OAD'])

    data.to_csv('bonds_not_clean.csv', index=False)

if __name__ == "__main__":
    main()
    