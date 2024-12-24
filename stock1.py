from datetime import datetime
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from backtest import backtest


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
def fetch_data(stock_code):
    """
    利用东方财富接口爬取股票历史数据
    :param stock_code: 股票代码（如 600519.SH）
    :return: 包含日期、价格、PE和分红的 pandas DataFrame
    """
    # 东方财富股票数据接口示例
    url = f"http://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        'secid': f'0.{stock_code}' if stock_code.endswith('.SH') else f'1.{stock_code}',
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56',
        'klt': 101,  # 日K线
        'fqt': 1,    # 前复权
        'beg': '20000101',  # 开始日期
        'end': '20500101',  # 结束日期
    }
    response = requests.get(url, params=params)
    data = response.json()

    if 'data' not in data or 'klines' not in data['data']:
        raise ValueError("无法获取数据，请检查股票代码或接口状态")

    # 解析数据
    records = data['data']['klines']
    parsed_data = []
    for record in records:
        fields = record.split(',')
        parsed_data.append({
            'Date': fields[0],
            'Price': float(fields[2]),  # 收盘价
            'PE': np.random.uniform(10, 30),  # 模拟市盈率（实际接口无此字段）
            'Dividend': np.random.uniform(0, 2),  # 模拟分红（实际接口无此字段）
        })

    # 转换为 DataFrame
    df = pd.DataFrame(parsed_data)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def plot_results(df, buy_signals, sell_signals):
    """
    绘制回测结果
    :param df: 包含回测结果的DataFrame
    :param buy_signals: 买入信号列表
    :param sell_signals: 卖出信号列表
    """
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Price'], label='Stock Price', color='blue')
    plt.plot(df.index, df['Portfolio Value'], label='Portfolio Value', color='green')

    # 标记买入和卖出信号
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

# 主程序
if __name__ == '__main__':
    stock_code = input("请输入股票代码（如 600519.SH）: ")

    # 获取数据
    data = fetch_data(stock_code)

    # 回测
    backtest_results, buy_signals, sell_signals = backtest(data)

    # 绘制结果
    plot_results(backtest_results, buy_signals, sell_signals)

    # 打印回测结果
    final_value = backtest_results['Portfolio Value'].iloc[-1]
    print(f"最终组合价值: {final_value:.2f} 元")