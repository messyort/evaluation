import matplotlib.pyplot as plt
import requests
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Set initial parameters
INITIAL_CAPITAL = 100000  # Initial investment amount
INVESTMENT_THRESHOLD_PE = 15  # PE threshold for buying
SELL_THRESHOLD_PE = 25  # PE threshold for selling
REINVEST_DIVIDENDS = True  # Whether to reinvest dividends

def generate_timestamp():
    """
    Generate a timestamp string in the format YYYYMMDDHHMMSS
    :return: Timestamp string
    """
    return datetime.now().strftime("%Y%m%d%H%M%S")

TIME_STAMPS = generate_timestamp()

def fetch_stock_data(stock_code):
    """
    Fetch historical stock data using Eastmoney API
    :param stock_code: Stock code (e.g., 600519.SH)
    :return: pandas DataFrame containing date, price, PE, and dividends
    """
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

    params = {
        'secid': f'1.{stock_code}' if stock_code.endswith('.SH') else f'0.{stock_code}',
        'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': 101,  # Daily K-line
        'fqt': 1,    # Forward adjusted
        'end': '20500101',  # End date
        'lmt': 5,  # Limit
    }
    response = requests.get(url, params=params)
    kLineData = response.json()
    # Save raw data to JSON file
    with open(f"{stock_code}_raw_data_{TIME_STAMPS}.json", "w") as json_file:
        json.dump(kLineData, json_file)
    print(kLineData)
    return kLineData
    # if 'data' not in kLineData or 'klines' not in kLineData['data']:
    #     raise ValueError("Unable to fetch data, please check the stock code or API status")
    
    

    # Parse data
    # records = kLineData['data']['klines']
    # parsed_data = []
    # for record in records:
    #     fields = record.split(',')
    #     parsed_data.append({
    #         'Date': fields[0],
    #         'Open': float(fields[1]),
    #         'Close': float(fields[2]),
    #         'High': float(fields[3]),
    #         'Low': float(fields[4]),
    #         'Volume': int(fields[5]),
    #         'Turnover': float(fields[6]),
    #         'Amplitude': float(fields[7]),
    #         'Change': float(fields[8]),
    #         'ChangePercent': float(fields[9]),
    #         'TurnoverRate': float(fields[10]),
    #         # 'PE': np.random.uniform(10, 30),  # Simulated PE ratio (actual API does not provide this field)
    #         # 'Dividend': np.random.uniform(0, 2),  # Simulated dividend (actual API does not provide this field)
    #     })

    # # Convert to DataFrame
    # df = pd.DataFrame(parsed_data)

    # df.set_index('Date', inplace=True)
    # return df

# Adjust the values in the fetch_financial_data accordingly
def adjust_values(record):
    record['f23'] = record['f23'] / 100  # Adjust Price-to-Book Ratio
    record['f37'] = record['f37'] / 100  # Adjust Dividend Yield
    record['f49'] = record['f49'] / 100  # Adjust ROE
    record['f129'] = record['f129'] / 100  # Adjust Debt-to-Equity Ratio
    record['f133'] = record['f133'] / 100  # Adjust Gross Margin
    record['f134'] = record['f134'] / 100  # Adjust Operating Margin
    return record
    
def fetch_financial_data(stock_code):
    """
    Fetch financial data using Eastmoney API
    :param stock_code: Stock code (e.g., 000858)
    :return: JSON response containing financial data
    """
    url = "https://push2delay.eastmoney.com/api/qt/slist/get"
    params = {
        'fltt': 1,
        'invt': 2,
        # 'cb': 'jQuery35104147345236595801_1735019044190',
        'fields': 'f12,f13,f14,f20,f58,f45,f132,f9,f152,f23,f49,f131,f137,f133,f134,f135,f129,f37,f1000,f3000,f2000',
        'secid': f'0.{stock_code}',
        'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
        'pn': 1,
        'np': 1,
        'spt': 1,
        'wbp2u': '|0|0|0|web',
        '_': '1735019044191'
    }
    response = requests.get(url, params=params)
    financial_data = response.json()
    
    # Save raw data to JSON file
    with open(f"{stock_code}_financial_data_{TIME_STAMPS}.json", "w") as json_file:
        json.dump(financial_data, json_file)
    
    print(financial_data)
    return financial_data

def handle_data(stock_code):
    financial_data = fetch_financial_data(stock_code)
    stock_data = fetch_stock_data(stock_code)
    
    if 'data' not in financial_data or 'diff' not in financial_data['data']:
        raise ValueError("Unable to fetch financial data, please check the stock code or API status")
    
    if 'data' not in stock_data or 'klines' not in stock_data['data']:
        raise ValueError("Unable to fetch stock data, please check the stock code or API status")
    
    financial_records = financial_data['data']['diff']
    stock_records = stock_data['data']['klines']
    
    financial_columns = [
    'Stock Code', 'Market', 'Stock Name', 'Market Cap', 'Total Assets', 'Net Profit', 
    'Revenue', 'PE Ratio', 'Shares Outstanding', 'Price-to-Book Ratio', 'ROE', 
    'Current Ratio', 'Gross Margin', 'Operating Margin', 'Net Income', 
    'Debt-to-Equity Ratio', 'Dividend Yield', 'EPS'
    ]
    
    stock_columns =  ['Date', 'Open', 'Close', 'High', 'Low', 'Volume', 'Turnover','Amplitude',
                      'Change','ChangePercent','TurnoverRate']
        # field_names = {
    #     'f9': 'PE Ratio',
    #     'f12': 'Stock Code',
    #     'f13': 'Market',
    #     'f14': 'Stock Name',
    #     'f20': 'Market Cap',
    #     'f23': 'Price-to-Book Ratio',
    #     'f37': 'Dividend Yield',
    #     'f45': 'Net Profit',
    #     'f49': 'ROE',
    #     'f58': 'Total Assets',
    #     'f129': 'Debt-to-Equity Ratio',
    #     'f131': 'Current Ratio',
    #     'f132': 'Revenue',
    #     'f133': 'Gross Margin',
    #     'f134': 'Operating Margin',
    #     'f135': 'Net Income',
    #     'f137': 'EPS',
    #     'f152': 'Shares Outstanding',
    # }
    parsed_financial_data = []
    for record in financial_records:
        # record = adjust_values(record)
        parsed_financial_data.append([
            record.get('f12', ''), record.get('f13', ''), record.get('f14', ''), 
            record.get('f20', ''), record.get('f58', ''), record.get('f45', ''), 
            record.get('f132', ''), record.get('f9', ''), record.get('f152', ''), 
            record.get('f23', ''), record.get('f49', ''), record.get('f131', ''), 
            record.get('f133', ''), record.get('f134', ''), record.get('f135', ''),
            record.get('f129', ''), record.get('f37', ''), record.get('f137', ''),      
        ])
    
    parsed_stock_data = []
    for record in stock_records:
        fields = record.split(',')
        parsed_stock_data.append(fields)
    
    df_financial = pd.DataFrame(parsed_financial_data, columns=financial_columns)
    df_stock = pd.DataFrame(parsed_stock_data, columns=stock_columns)
    
    # Combine the dataframes and remove duplicate columns
    combined_df = pd.concat([df_stock, df_financial], axis=1)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    
    return combined_df

def export_to_excel(df, stock_code):
    """
    Export data to Excel file
    :param df: DataFrame containing price, PE, and dividends
    :param stock_code: Stock code
    """
    file_name = f"{stock_code}_{TIME_STAMPS}.xlsx"
    df.to_excel(file_name)
    print(f"Data exported to {file_name}")

def backtest(df):
    """
    Perform historical backtest based on the model
    :param df: DataFrame containing price, PE, and dividends
    :return: Account balance history, investment position history
    """
    cash = INITIAL_CAPITAL
    shares = 0
    portfolio_value = []
    cash_balance = []
    buy_signals = []
    sell_signals = []

    for date, row in df.iterrows():
        price = row['Price']
        pe = row['PE']
        dividend = row['Dividend']

        # Reinvest dividends
        if REINVEST_DIVIDENDS and shares > 0:
            cash += shares * dividend

        # Buy signal
        if pe <= INVESTMENT_THRESHOLD_PE and cash >= price:
            shares_to_buy = cash // price
            shares += shares_to_buy
            cash -= shares_to_buy * price
            buy_signals.append((date, price))
        # Sell signal
        elif pe >= SELL_THRESHOLD_PE and shares > 0:
            cash += shares * price
            shares = 0
            sell_signals.append((date, price))

        # Record portfolio value
        portfolio_value.append(cash + shares * price)
        cash_balance.append(cash)

    df['Portfolio Value'] = portfolio_value
    df['Cash Balance'] = cash_balance
    return df, buy_signals, sell_signals

def plot_results(df, buy_signals, sell_signals):
    """
    Plot backtest results
    :param df: DataFrame containing backtest results
    :param buy_signals: List of buy signals
    :param sell_signals: List of sell signals
    """
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Price'], label='Stock Price', color='blue')
    plt.plot(df.index, df['Portfolio Value'], label='Portfolio Value', color='green')

    # Mark buy and sell signals
    for signal in buy_signals:
        plt.scatter(signal[0], signal[1], marker='^', color='red', label='Buy Signal')
    for signal in sell_signals:
        plt.scatter(signal[0], signal[1], marker='v', color='orange', label='Sell Signal')

    plt.title('Investment Backtest')
    plt.xlabel('Date')
    plt.ylabel('Price / Portfolio Value')
    plt.legend()
    plt.grid()
    plt.show()

# Main program
if __name__ == '__main__':
    stock_code = input("Please enter the stock code (e.g., 600519.SH): ")

    # Handle data
    # data1 = fetch_stock_data(stock_code)
    # data2 = fetch_financial_data(stock_code)
    data = handle_data(stock_code)

    # Export data to Excel
    export_to_excel(data, stock_code)

    # Backtest
    # backtest_results, buy_signals, sell_signals = backtest(data)

    # Plot results
    # plot_results(backtest_results, buy_signals, sell_signals)

    # Print backtest results
    # final_value = backtest_results['Portfolio Value'].iloc[-1]
    # print(f"Final portfolio value: {final_value:.2f} CNY")