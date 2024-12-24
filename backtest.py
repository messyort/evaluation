def backtest(df):
    """
    根据模型进行历史回测（分批买卖逻辑）
    :param df: 包含价格、PE和分红的DataFrame
    :return: 账户余额历史、投资持仓历史
    """
    cash = 1000000  # 初始现金
    shares = 0
    portfolio_value = []
    cash_balance = []
    buy_signals = []
    sell_signals = []

    BUY_PERCENTAGE = 0.2  # 每次买入的现金比例（例如 20%）
    SELL_PERCENTAGE = 0.2  # 每次卖出的股票比例（例如 20%）
    INVESTMENT_THRESHOLD_PE = 15  # 买入阈值 PE
    SELL_THRESHOLD_PE = 25  # 卖出阈值 PE
    REINVEST_DIVIDENDS = True  # 是否分红再投资

    for date, row in df.iterrows():
        price = row['Price']
        pe = row['PE']
        dividend = row['Dividend']

        # 分红再投资
        if REINVEST_DIVIDENDS and shares > 0:
            cash += shares * dividend

        # 分批买入逻辑
        if pe <= INVESTMENT_THRESHOLD_PE and cash >= price:
            # 买入固定比例的现金对应的股票
            investment_amount = cash * BUY_PERCENTAGE
            shares_to_buy = investment_amount // price
            if shares_to_buy > 0:
                shares += shares_to_buy
                cash -= shares_to_buy * price
                buy_signals.append((date, price))

        # 分批卖出逻辑
        elif pe >= SELL_THRESHOLD_PE and shares > 0:
            # 卖出固定比例的持仓
            shares_to_sell = int(shares * SELL_PERCENTAGE)
            if shares_to_sell > 0:
                cash += shares_to_sell * price
                shares -= shares_to_sell
                sell_signals.append((date, price))

        # 记录组合价值
        portfolio_value.append(cash + shares * price)
        cash_balance.append(cash)

    df['Portfolio Value'] = portfolio_value
    df['Cash Balance'] = cash_balance
    return df, buy_signals, sell_signals
