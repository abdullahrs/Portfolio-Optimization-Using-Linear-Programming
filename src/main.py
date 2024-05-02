import pandas as pd
import numpy as np
import yfinance as yf
import cvxpy as cp


import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime, timedelta


class PortfolioOptimizer:
    
    def __init__(self) -> None:
        # Current date
        self.end_date = datetime.now()

        # Start date set to 15 years before the end date
        self.start_date = self.end_date - timedelta(days=365*15)  # Approximating 365 days per year

        # Defining portfolio components 
        self.item_symbols = {'BIST100': 'XU100.IS', 'Gold': 'GC=F', 'Silver': 'SI=F', 'USDTRY': 'USDTRY=X'}
        
        self.returns = {}
        
    def get_historical_data(self, portfolio_item: str, start_date: datetime, end_date: datetime):
        """
        Fetches historical market data from Yahoo Finance.
        """
        data = yf.download(portfolio_item, start=start_date, end=end_date)
        return data['Adj Close']
    
    def calculate_annualized_returns(self, data):
        """
        Calculates annualized logarithmic returns from daily prices.
        """
        daily_returns = np.log(data / data.shift(1))
        # ~252 trading days in a year
        annualized_returns = daily_returns.mean() * 252
        return annualized_returns
    
    def calculate_annualized_log_returns(self, dataframe, date_col, value_col, optional_date_format : str = None):
        """
        Used for house price index and interest rate instruments
        Calculate the annualized logarithmic returns from a time series data.
        Args:
        dataframe (pd.DataFrame): DataFrame containing the data.
        date_col (str): Column name for dates.
        value_col (str): Column name for values (e.g., HPI or asset prices).
        optional_date_format (bool): Variable for format warnings.
        
        Returns:
        float: Annualized logarithmic return.
        """
        # Convert date column to datetime and ensure data is sorted
        dataframe[date_col] = pd.to_datetime(dataframe[date_col], format= optional_date_format if optional_date_format != None else None)
        dataframe.sort_values(by=date_col, inplace=True)

        # Convert the value column to float to avoid type issues
        dataframe[value_col] = pd.to_numeric(dataframe[value_col], errors='coerce')

        # Drop rows with NaN values that might disrupt calculations
        dataframe.dropna(subset=[value_col], inplace=True)

        # Calculate logarithmic returns
        dataframe['log_returns'] = np.log(dataframe[value_col] / dataframe[value_col].shift(1))

        # Calculate annualized return (assuming the data is monthly, multiply by 12)
        annualized_return = dataframe['log_returns'].mean() * 12
        return annualized_return
    
    def load_and_prepare_interest_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Load and prepare interest data from a transposed Excel file.
        Args:
        filepath (str): Path to the Excel file.
        
        Returns:
        pd.DataFrame: Prepared DataFrame with 'Date' as a column.
        """
        # Load data, transpose and reset index to make the index a column
        df = dataset.copy().T
        df.reset_index(inplace=True)
        df.columns = df.iloc[0]  # Set the first row as column header
        df = df[1:]  # Remove the first row after setting headers
        df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)  # Rename the date column
        # df.set_index('Date', inplace=True)
        return df
    
    def preprocessor(self):
        # Fetch data
        self.historical_data = {asset: self.get_historical_data(item, self.start_date, self.end_date) for asset, item in self.item_symbols.items()}
        # Load House Price Index & Historical Interest Rate Data
        self.hpi_df = pd.read_excel('./data/konut-fiyat-endeks.xlsx')
        self.interest_df = pd.read_excel('./data/tcmb-faiz.xlsx')
        self.interest_df = self.load_and_prepare_interest_data(self.interest_df)
        self.historical_data['HPI'] = pd.Series(data=self.hpi_df['TP HKFE01'].values, index=self.hpi_df['Tarih'])
        self.historical_data['INT'] = pd.Series(data=self.interest_df['1 Yıla Kadar Vadeli (TL)(%)'].values, index=self.interest_df['Date'])
    
    def calculate_returns(self):
        # Calculate annualized returns for each asset
        for asset, data in self.historical_data.items():
            if asset == 'HPI':
                self.returns[asset] = self.calculate_annualized_log_returns(self.hpi_df, 'Tarih', 'TP HKFE01')
            elif asset == 'INT':
                self.returns[asset] = self.calculate_annualized_log_returns(self.interest_df, 'Date', '1 Yıla Kadar Vadeli (TL)(%)', optional_date_format='%Y-%m')
            else:
                self.returns[asset] = self.calculate_annualized_returns(data)
                
    def currency_conversion(self):
        self.historical_data['Gold'] = self.historical_data['Gold'] * self.historical_data['USDTRY']
        self.historical_data['Silver'] = self.historical_data['Silver'] * self.historical_data['USDTRY']
        # Convert the series into DataFrames if necessary
        bist100_df = self.historical_data['BIST100'].to_frame(name='BIST100')
        usdtry_df = self.historical_data['USDTRY'].to_frame(name='USDTRY')

        # Join the dataframes on their index, which are the dates
        combined_data = bist100_df.join(usdtry_df, how='inner')

        # Calculate the BIST100 in USD
        combined_data['BIST100_USD'] = combined_data['BIST100'] / combined_data['USDTRY']
        
        # Forward fill missing values in USDTRY data
        combined_data['USDTRY'].fillna(method='ffill', inplace=True)

        # Recalculate BIST100 in USD after filling
        combined_data['BIST100_USD'] = combined_data['BIST100'] / combined_data['USDTRY']
        
        # Display the first few rows of the combined data to check
        print(combined_data.head())
        print(combined_data.tail())

        # plt.figure(figsize=(12, 6))
        # plt.plot(combined_data['BIST100_USD'], label='BIST 100 in USD')
        # plt.title('BIST 100 Indexed to USD Over Time')
        # plt.xlabel('Date')
        # plt.ylabel('Indexed Value')
        # plt.legend()
        # plt.show()




        

    def exec(self):
        self.preprocessor()
        self.calculate_returns()
        self.currency_conversion()
        
        
if __name__ == "__main__":
    PortfolioOptimizer().exec()