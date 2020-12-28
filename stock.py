import datetime as dt
import pandas as pd
import pandas_datareader.data as web

def get_yahoo_data(start,end):
    df = web.DataReader('TSLA', 'yahoo', start, end)
    print(df.tail())

start = dt.datetime(2020, 9, 1)
end = dt.datetime(2020, 12, 25)
get_yahoo_data(start,end)

#read https://einvestingforbeginners.com/yahoo-finance-beginners-guide/

