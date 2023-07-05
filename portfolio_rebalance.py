import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import copy


# Download historical data (monthly) for DJI constituent stocks

tickers = ["AAPL","GOOG","MSFT","AMZN","TSLA","NVDA","JPM","BAC","WFC","UNH"]

ohlc_week = {} # directory with ohlc value for each stock            
start = dt.datetime.today()-dt.timedelta(365*4)
end = dt.datetime.today()

# looping over tickers and creating a dataframe with close prices
for ticker in tickers:
    ohlc_week[ticker] = yf.download(ticker,start,end,interval='1wk')
    ohlc_week[ticker].dropna(inplace=True,how="all")
 
#tickers = ohlc_mon.keys() # redefine tickers variable after removing any tickers with corrupted data

def ATR(DF, n=14):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Adj Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Adj Close"].shift(1))
    df["TR"] = df[["H-L","H-PC","L-PC"]].max(axis=1, skipna=False)
    df["ATR"] = df["TR"].ewm(com=n, min_periods=n).mean()
    return df["ATR"]
atr_df = pd.DataFrame()
for ticker in tickers:
    atr_df[ticker] = ATR(ohlc_week[ticker])
    atr_df[ticker].dropna(inplace=True,how="any")
    atr_df[ticker] = atr_df[ticker][52:]
    
################################Backtesting####################################

# calculating monthly return for each stock and consolidating return info by stock in a separate dataframe
ohlc_dict = copy.deepcopy(ohlc_week)
return_df = pd.DataFrame()
for ticker in tickers:
    print("calculating weekly rolling return for ",ticker)
    ohlc_dict[ticker]["week_ret"] = ohlc_dict[ticker]["Adj Close"].pct_change()
    ohlc_dict[ticker]["rweek_ret"] = ohlc_dict[ticker]["week_ret"].rolling(window=52).sum()
    return_df[ticker] = ohlc_dict[ticker]["rweek_ret"]
return_df.dropna(inplace=True)


# function to calculate portfolio return iteratively
def pflio(DF,x=5):
    """Returns cumulative portfolio return
    DF = dataframe with weekly return info for all stocks
    m = number of stock in the portfolio
    x = number of stocks to present in the portfolio"""
    df = DF.copy()
    portfolio =[]
    weekly_ret = []
    for i in range(0,len(df)):
        best_list = df.iloc[i,:].sort_values(ascending=False)[:5].index.values.tolist()
        portfolio= portfolio + best_list
        print(best_list)
        weeklyti_ret=0
        invested = 0
        for ticker in best_list:
            current_price = ohlc_dict[ticker]['Close'].iloc[i]
            open_price = ohlc_dict[ticker]['Open'].iloc[i]
            stop_loss = open_price - (2 * atr_df[ticker][i])
            invested += ohlc_dict[ticker]['Open'].iloc[i]
            if current_price > open_price:
                weeklyti_ret += current_price - open_price

        # Check if stop loss is hit
            if current_price <= stop_loss:
                weeklyti_ret += stop_loss - open_price
    
        weekly_ret.append((weeklyti_ret/invested))
    weekly_ret_df = pd.DataFrame(np.array(weekly_ret),columns=["week_ret"])
    return weekly_ret_df

weekly_return = pflio(return_df,5)



#calculating overall strategy's KPIs
CAGR(pflio(return_df,6,3))
sharpe(pflio(return_df,6,3),0.025)
max_dd(pflio(return_df,6,3)) 

#calculating KPIs for Index buy and hold strategy over the same period
DJI = yf.download("^DJI",dt.date.today()-dt.timedelta(3650),dt.date.today(),interval='1mo')
DJI["mon_ret"] = DJI["Adj Close"].pct_change().fillna(0)
CAGR(DJI)
sharpe(DJI,0.025)
max_dd(DJI)

#visualization
fig, ax = plt.subplots()
plt.plot((1+pflio(return_df,6,3)).cumprod())
plt.plot((1+DJI["mon_ret"].reset_index(drop=True)).cumprod())
plt.title("Index Return vs Strategy Return")
plt.ylabel("cumulative return")
plt.xlabel("months")
ax.legend(["Strategy Return","Index Return"])