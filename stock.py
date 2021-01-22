import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

style.use('ggplot')

start=dt.datetime(2020,1,1)
end=dt.datetime(2021,1,2)

def get_yahoo_data(start,end):
    df = web.DataReader('TSLA', 'yahoo', start, end)
    df.to_csv('tsla.csv')
    print(df.tail())

def read_csv(f):
    df=pd.read_csv(f,parse_dates=True,index_col=0)
    print (df.head())

def get_holders():
#https://towardsdatascience.com/a-comprehensive-guide-to-downloading-stock-prices-in-python-2cd93ff821d4
    Ticker = yf.Ticker('TSLA')
    tsla_df = ticker.institutional_holders
    print (tsla_df.head())
    tsla_df.to_csv('C:/Users/encylin/PycharmProjects/finance/tsla.csv')


#get_yahoo_data(start,end)
read_csv('tsla.csv')

#read https://einvestingforbeginners.com/yahoo-finance-beginners-guide/

#1 y target Est.


#to read:  https://towardsdatascience.com/time-series-forecasting-in-real-life-budget-forecasting-with-arima-d5ec57e634cb
