'''
MACD
2011年〜2018年の最適な組み合わせを見つける。
'''
import os
import datetime
import pandas as pd
import pandas_datareader
import matplotlib.pyplot as plt
import japanize_matplotlib
from mpl_toolkits.mplot3d import Axes3D
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, plot_heatmaps

START_DATE = datetime.date(2011, 1, 1)
END_DATE = datetime.date(2018, 12, 31)
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
    fastperiod = 12
    slowperiod = 26
    signalperiod = 9

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

    stats, heatmap = bt.optimize(
        fastperiod=range(5, 51),
        slowperiod=range(5, 51),
        signalperiod=range(5, 51),
        return_heatmap=True,
        constraint=lambda p: p.fastperiod < p.slowperiod)

    print(stats)
    print(stats['_strategy'])
    bt.plot()
    plot_heatmaps(heatmap, agg='mean', plot_width=2048, filename='heatmap')

    # 3Dヒートマップで表示
    d = heatmap.reset_index()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(d['fastperiod'], d['slowperiod'], d['signalperiod'], c=d['SQN'])
    ax.set_xlabel('Fast Period')
    ax.set_ylabel('Slow Period')
    ax.set_zlabel('Signal Period')
    plt.show()

    # 上記だとデータが多すぎるので、SQN > 1のデータのみを表示する
    d2 = d[d['SQN'] > 1]
    fig = plt.figure()
    ax = Axes3D(fig)
    mappable = ax.scatter(
        d2['fastperiod'], d2['slowperiod'], d2['signalperiod'], c=d2['SQN'])
    fig.colorbar(mappable, ax=ax)
    ax.set_xlabel('Fast Period')
    ax.set_ylabel('Slow Period')
    ax.set_zlabel('Signal Period')
    plt.show()


if __name__ == '__main__':
    main()
