import numpy as np
import pandas as pd
import yfinance
import yfinance as yf
import matplotlib.pyplot as plt

#Download OHLCV Data
tickers =['A17U','N2IU']
tickers = [ticker+'.SI' for ticker in tickers]

data = yfinance.download(tickers=tickers) #period to get specific range e.g. '1mo' for last 1 month data
data = data.drop(columns=['Adj Close', 'High', 'Low', 'Volume', 'Open'])

mask = data[('Close', 'N2IU.SI')].notna().to_list()
n2iu_listing_date = data[('Close', 'N2IU.SI')][mask].idxmin()
data = data[n2iu_listing_date:]

data[('Returns','A17U.SI')] = data.pct_change()[('Close','A17U.SI')]
data[('Returns','N2IU.SI')] = data.pct_change()[('Close','N2IU.SI')]

#Split into 2 df


#Get annualised return
#df_returns = data.Returns.groupby(data.index.to_period('M')).last()
df_returns = data.Returns
num_days = df_returns.shape[0]-1
a17u_DannR = ((df_returns['A17U.SI']+1).prod()**(252/num_days))-1
print('===DAILY===')
print(f'A17U Daily Annualised Return: {(a17u_DannR*100).round(2)}')
a17u_DannVol = df_returns['A17U.SI'].std()*np.sqrt(252)
print(f'A17U Daily Annualised Volatility: {a17u_DannVol}')

ann_risk_free_rate = 0.03

sharpeD = (a17u_DannR-ann_risk_free_rate)/a17u_DannVol
print(f'Sharpe Ratio Daily: {sharpeD}\n')

#Monthly frequency of observation
df_returnsM = data.Close[['A17U.SI']]
df_returnsM.index = df_returnsM.index.to_period('M')
df_returnsM = df_returnsM.groupby(df_returnsM.index).last()
df_returnsM['Returns'] = df_returnsM.pct_change()
df_returnsM = df_returnsM.dropna()['Returns']
num_months = df_returnsM.shape[0]
a17u_MannR = ((1+df_returnsM).prod())**(12/num_months)-1

a17u_MannVol = df_returnsM.std()*np.sqrt(12)
print('===MONTHLY===')
print(f'A17U Monthly Annualised Return: {(a17u_MannR*100).round(2)}')
print(f'A17U Monthly Annualised Volatility: {a17u_MannVol}')
sharpeM = (a17u_MannR-ann_risk_free_rate)/a17u_MannVol
print(f'Sharpe Ratio Monthly: {sharpeM}\n')

#Calculate drawdown monthly frequency of observation
#wealth index
#previous peaks
#calmar last 36 months
#drawdown
wealth_index = 1000*((df_returnsM.iloc[-37:]+1).cumprod())
previous_peaks = wealth_index.cummax()

wealth_index.plot()
previous_peaks.plot()


drawdown = (wealth_index-previous_peaks)/previous_peaks
#drawdown.iloc[-37:].plot()
max_drawdown = drawdown.min()
print(f'Max drawdown: {drawdown.min()} occuring in {max_drawdown}')

#calmar ratio of trailing 36 months
a17u_calmar = a17u_MannR/max_drawdown
print(f'Calmar Ratio of  A17U: {a17u_calmar}')

plt.show()







