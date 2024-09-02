import numpy as np
import pandas as pd
import edhec_risk_kit as erk
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#2-Asset Efficient frontier

# 1. Load data
ind = pd.read_csv('data/ind30_m_vw_rets.csv', index_col=0)/100
ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
ind.columns = ind.columns.str.strip() #Strips whitespace from headers

# 2. Plot drawdown
drawdown = erk.drawdowns(ind.loc[:,['Food','Beer','Smoke']])
print(drawdown)
#drawdown.plot.line(title='Drawdowns of Food industry', subplots=True, figsize=(12,6), xlabel='Year', legend=True)
#plt.show()

# 3. Get Cornish-Fisher modified VaR
mod_VaR = ind.loc[:,['Food','Beer','Smoke']].agg(erk.cornish_fisher_modifiedVaR, level=0.05)
#print(mod_VaR)

#all_mod_VaR = ind.agg(erk.cornish_fisher_modifiedVaR, level=0.05).sort_values().plot.bar(title='Modified VaR for all industries')
#plt.show()

#4. Observe Sharpe Ratio across assets
#sharpeBar = ind['1995':'2000'].agg(erk.sharpe_ratio, riskfree_rate=0.03, periods_per_year=12).sort_values(ascending=True).plot(kind='bar')
#plt.show()

# #Get covariance matrix. Diagonals are the variance. The rest are covariance.
# cov = ind['1995':'2000'].cov()
#
# ### Weighted Portfolio Return and Volatility (Return vs. Risk)
# ind = ind
# cov = ind['1996':'2000'].cov()
# er = erk.annualize_rets(ind['1996':'2000'], 12) #expected return
# #print(er)
#
#
#
# l = ['Food','Beer','Smoke','Coal']
# #er = er[l] #Indexing a series must pass in a list of columns
# # cov = cov.loc[l,l]
# # w = np.repeat(0.25, 4) #4 asset each 1/4
#
# #port_return = erk.portfolio_return(weights=w, returns=er)
# #port_vol = erk.portfolio_vol(weights=w, cov=cov)
# #print(f'The equally weighted portfolio of Food,Beer,Smoke,Coal gives return {port_return} and has volatility of {port_vol}.')


## 2-Asset Frontier
l = ['Games', 'Fin']
er = erk.annualize_rets(ind['1996':'2000'], 12) #expected return
er = er[l]
cov = ind['1996':'2000'].cov()
cov = cov.loc[l,l]

#print(erk.ef_2asset(n_points=20,er=er, cov=cov))

## n-Asset Frontier
# Give list of returns -> output weights that give least volatility
# pass the weights in to plot eff frontier

#optimum wieght for return of 15%
w15 = erk.minimize_vol(target_return=0.15, er=er, cov=cov)
vol15 = erk.portfolio_vol(w15, cov)

num_points = 20
#Get list of weight based on min and max returns
w_optimum = [erk.minimize_vol(r, er=er, cov=cov) for r in np.linspace(er.min(), er.max(), num=num_points)]
er_list = [erk.portfolio_return(w, er) for w in w_optimum]
vol_list = [erk.portfolio_vol(w, cov) for w in w_optimum]
ef_df = pd.DataFrame({
    'er':er_list,
    'vol':vol_list
})
ef_df.loc[:,['er','vol']].plot(kind='line', x='vol', y='er', style='.-')
plt.show()