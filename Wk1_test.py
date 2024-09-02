import numpy as np
import pandas as pd
import edhec_risk_kit as erk
import scipy.stats.stats as st
from scipy.stats import norm
import matplotlib.pyplot as plt


file = 'data/Portfolios_Formed_on_ME_monthly_EW.csv'
#Load file, rename date col, set date col as index
data = pd.read_csv(file).rename(columns={'Unnamed: 0':'Date'}).loc[:,['Date','Lo 20', 'Hi 20']]

data.Date = pd.to_datetime(data.Date, format='%Y%m')
data.index = data.Date
data.drop(columns='Date', inplace=True)
data.index = data.index.to_period("M")

returns = data/100

annualised_ret = returns.agg(erk.annualised_return)

print(f'Annualised returns: \n{annualised_ret}\n')

annualised_vol = returns.agg(erk.annualised_volatility)

print(f'Annualised volatility: \n{annualised_vol}\n')

#1999-2015
annualised_ret = returns.loc['1999':'2015',:].agg(erk.annualised_return)

print(f'Annualised returns 1999-2015: \n{annualised_ret}\n')

annualised_vol = returns.loc['1999':'2015',:].agg(erk.annualised_volatility)

print(f'Annualised volatility 1999-2015: \n{annualised_vol}\n')

#Drawdown calculation for Lo 20
drawdown_lo20_1999_2015 = erk.drawdowns(returns.loc['1999':'2015','Lo 20'])

print(f'Max Drawdown of Lo20 1999-2015: {-drawdown_lo20_1999_2015.min()} occuring on: {drawdown_lo20_1999_2015.idxmin()}')

#Drawdown calculation for Hi 20
drawdown_hi20_1999_2015 = erk.drawdowns(returns.loc['1999':'2015','Hi 20'])

print(f'Max Drawdown of Hi20 1999-2015: {-drawdown_hi20_1999_2015.min()} occuring on: {drawdown_hi20_1999_2015.idxmin()}\n')


#Part 2
# Load file
data2 = 'data/edhec-hedgefundindices.csv'
hfi = pd.read_csv(data2, index_col='date', header=0)
hfi = hfi/100
hfi.index = pd.to_datetime(hfi.index, infer_datetime_format=True).to_period('M')

hfi2009_2018 = hfi.loc['2009':'2018',:]
#Semideviation
print(f'Semideviations 2019-2018: \n{hfi2009_2018.agg(erk.semideviation).sort_values(ascending=False)}\n')

#Skewness
print(f'Skewness 2019-2018: \n{hfi2009_2018.agg(erk.skewness).sort_values(ascending=False)}\n')

#Kurtosis
hfi2000_2018 = hfi.loc['2000':'2018',:]

print(f'Kurtosis 2000-2018: \n{hfi2000_2018.agg(erk.kurtosis).sort_values(ascending=False)}\n')

#historical VaR
print(f'Historical VaR: \n{hfi.agg(erk.historicalVaR)}\n')

#Conditional Historic VaR
print(f'Conditional Historic VaR: \n{hfi.agg(erk.conditionalVaR)}\n')

#Parametric Gaussian VaR
print(f'Parametric Gaussian VaR: \n{hfi.agg(erk.parametric_gaussianVaR)}\n')

#Cornish-Fisher Modified Var
print(f'Cornish-Fisher Modified VaR: \n{hfi.agg(erk.cornish_fisher_modifiedVaR)}\n')

#JB test for nomality
print(f'Are the distributions normally distributed?: \n{hfi.agg(erk.is_normal)}\n')

