from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import requests
import matplotlib.pyplot as plt
from pprint import pprint
import json

class StockAnalyzer:

    def __init__(self):
        self.quote = -999
        self.ticker = ''
        self.high = 999
        self.low = -999
        self.yahoo_clientId = 'dj0yJmk9VGpjTXhUUDZCcDFZJmQ9WVdrOVRXMU9aRTFsTjJNbWNHbzlNQS0tJnM9Y29uc3VtZXJzZWNyZXQmeD03MQ--'
        self.yahoo_clientSecret = '6768cc45197e6de122d2bc3b4e3ef8fcd6a5ee1b'
        self.alphaVantage_Appid = 'YD1VBM3EOR8EXKEC'

    def GetStockInformation(self, Ticker):
        self.ticker = Ticker
        ts = TimeSeries(key=self.alphaVantage_Appid, output_format='pandas')
        ti = TechIndicators(key=self.alphaVantage_Appid, output_format='pandas')
        data, meta_data = ts.get_intraday(symbol=Ticker,interval='1min',  outputsize='full')
        indicators, indicator_metadata = ti.get_sma(symbol=Ticker,interval='1min', time_period=100, series_type='close')
        #data['close'].plot()
        #indicators.plot()
        #plt.show()

        latest = len(data)-1
        iLatest = len(indicators)-1
        
        print(data['close'][latest])
        print(indicators['SMA'][iLatest])
        if data['close'][latest] > indicators['SMA'][iLatest]:
            print('High')
        else:
            print('Low')

    def GetIExtradingStockInfo(self, Ticker):
        response = requests.get(url='https://api.iextrading.com/1.0/stock/'  + Ticker + '/stats')
        t = json.loads(response.text)
        pprint(t)
        #response = requests.get('')

    def GetStockNews(self, Ticker):
        print('not implemented')