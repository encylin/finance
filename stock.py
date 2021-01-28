import datetime as dt
import os
import matplotlib.pyplot as plt
from matplotlib import style
import mplfinance as mpf
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader.data as web
import bs4 as bs
import pickle    #serialize any python object
import requests
import numpy as np
import yfinance as yf

style.use('ggplot')
start=dt.datetime(2018,1,1)
end=dt.datetime(2021,1,25)

def get_yahoo_data(start,end):
    df = web.DataReader('TSLA', 'yahoo', start, end)
    df.to_csv('tsla.csv')
    print(df.tail())

def xy():
    ax1 = plt.subplot2grid((6,1),(0,0),rowspan=5,colspan=1)   #grid size  start point,
    ax2 = plt.subplot2grid((6,1),(5,0),rowspan=1,colspan=1,sharex=ax1)
    return ax1, ax2

def read_rolling(f):
    df=pd.read_csv(f,parse_dates=True,index_col=0)
    #df['100ma']=df['Adj Close'].rolling(window=100).mean()
    #df.dropna(inplace=True)      # handle 1st 99 no value rows
    df['100ma']=df['Adj Close'].rolling(window=100,min_periods=0).mean()
    # df['Adj Close'].plot()
    ax1, ax2= xy()
    ax1.plot(df.index,df['Adj Close'])
    ax1.plot(df.index,df['100ma'])
    ax2.bar(df.index,df['Volume'])
    plt.show()

def read_candel(f):
    df = pd.read_csv(f, parse_dates=True, index_col=0)
    # re-assmble data
    df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)
    df_ohlc = df['Adj Close'].resample('10D').ohlc()
    df_ohlc['volume']= df['Volume'].resample('10D').sum()
    print (df_ohlc.head())
 #   mpf.plot(df_ohlc, type='candle')
    mpf.plot(df_ohlc, type='candle', mav=(3, 6, 9),volume=True)
    #df_ohlc.reset_index(inplace=True)  # move date as the col index
    #df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)  # dates to plt format

def get_holders():
#https://towardsdatascience.com/a-comprehensive-guide-to-downloading-stock-prices-in-python-2cd93ff821d4
    Ticker = yf.Ticker('TSLA')
    tsla_df = ticker.institutional_holders
    print (tsla_df.head())
    tsla_df.to_csv('C:/Users/encylin/PycharmProjects/finance/tsla.csv')

def drawing():
    ax1 = plt.subplot2grid((6,1),(0,0),rowspan=5,colspan=1)   #grid size  start point,
    ax2 = plt.subplot2grid((6,1),(5,0),rowspan=1,colspan=1,sharex=ax1)
    ax1.plot(df.index,df['Adj Close'])
    ax1.plot(df.index,df['100ma'])
    ax2.bar(df.index,df['Volume'])
    plt.show()
    #df['Adj Close'].plot()
    #plt.show()

def save_sp500_tickers():
    resp=requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup=bs.BeautifulSoup(resp.text,'lxml')  # text of the source code
    table=soup.find('table',{'class':'wikitable sortable'})
    tickers=[]
    for row in table.findAll('tr')[1:]:
        ticker=row.findAll('td')[0].text.rstrip()
        if '.' in ticker:
            ticker=ticker.replace('.', '-')
        tickers.append(ticker)
    with open ("sp500tickers.pickle","wb") as f:  # write bits
        pickle.dump(tickers,f)
    print (tickers)
    return tickers

def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers=save_sp500_tickers()
    else:
        with open ("sp500tickers.pickle","rb") as f:
            tickers=pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    for ticker in tickers:
  #      print (ticker)
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            try:
                df=web.DataReader(ticker,'yahoo',start,end)
                df.to_csv('stock_dfs/{}.csv'.format(ticker))
            except:
                print ('{} cant be yahoo located'.format(ticker))
        else:
            print ("Already have {}".format(ticker))

def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df=pd.DataFrame()

    for count,ticker in enumerate(tickers):
        try:
            df=pd.read_csv('stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date',inplace=True)
            df.rename(columns={'Adj Close':ticker},inplace=True)
            df.drop(['Open','High','Low','Close','Volume'],1,inplace=True)
            if main_df.empty:
                main_df=df
            else:
                main_df=main_df.join(df,how='outer')
            if count % 10 == 0:
                print (count)
        except:
            print("missing {}".format(ticker))
    print (main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')

def visualize_data():
    df=pd.read_csv('sp500_joined_closes.csv')
#    df['AAPL'].plot()
 #   plt.show()
    df_corr=df.corr()
    print (df_corr.head())

    data=df_corr.values
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)

    heatmap=ax.pcolor(data,cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0])+0.5,minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels=df_corr.columns
    row_labels=df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_xticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    plt.show()


#visualize_data()
#compile_data()
#save_sp500_tickers()
#get_data_from_yahoo()
#read_candel('TSLA.csv')
#read_rolling('TSLA.csv')
#get_yahoo_data(start,end)
#read https://einvestingforbeginners.com/yahoo-finance-beginners-guide/
#1 y target Est.
#to read:  https://towardsdatascience.com/time-series-forecasting-in-real-life-budget-forecasting-with-arima-d5ec57e634cb
