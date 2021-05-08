'''
MACD
2011年〜2018年の最適な組み合わせを使って2019年〜2020年の取引を行う。
'''
import os
import datetime
import pandas as pd
import pandas_datareader
import matplotlib.pyplot as plt
import japanize_matplotlib
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

START_DATE = datetime.date(2019, 1, 1)
END_DATE = datetime.date(2020, 12, 31)
TICKER = '^N225'
INIT_CASH = 1000000


def get_stock(ticker, start_date, end_date):
    '''
    get stock data from Yahoo Finance
    '''
    dirname = '../data'
    os.makedirs(dirname, exist_ok=True)
    period = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    fname = f'{dirname}/{ticker}_{period}.pkl'
    if os.path.exists(fname):
        df_stock = pd.read_pickle(fname)
    else:
        df_stock = pandas_datareader.data.DataReader(
            ticker, 'yahoo', start_date, end_date)
        df_stock.to_pickle(fname)
    return df_stock


class MacdStrategy(Strategy):
    '''
    MACD Strategy
    '''
    fastperiod = 9
    slowperiod = 48
    signalperiod = 5

    def init(self):
        close = self.data['Adj Close']
        self.macd, self.macdsignal, _ = self.I(
            talib.MACD, close,
            fastperiod=self.fastperiod,
            slowperiod=self.slowperiod,
            signalperiod=self.signalperiod)

    def next(self):
        '''
        MACDとMACDシグナルのゴールデンクロスで買い
        MACDとMACDシグナルのデッドクロスで売り
        MACDがマイナスからプラスになったら買い
        MACDがプラスからマイナスになったら売り
        '''
        if crossover(self.macd, self.macdsignal):
            self.buy()
        elif crossover(self.macd, self.macdsignal):
            self.sell()
        elif crossover(self.macd, 0):
            self.buy()
        elif crossover(0, self.macd):
            self.sell()


def main():
    df = get_stock(TICKER, START_DATE, END_DATE)

    bt = Backtest(
        df,
        MacdStrategy,
        cash=INIT_CASH,
        trade_on_close=False,
        exclusive_orders=True
    )

    output = bt.run()
    print(output)
    bt.plot()


if __name__ == '__main__':
    main()
