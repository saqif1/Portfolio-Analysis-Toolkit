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
# Give list of returns -> output weights that give least volatility (minimize volatility)
# pass the weights in to plot eff frontier


def minimize_vol(target_return, er, cov):
    def sum_weights_is_1(w):
        return np.sum(w) - 1

    def return_is_target(w, er, target_return):
        return erk.portfolio_return(w, er) - target_return

    n = er.shape[0] #for repetition of other parameters
    args=(cov,),
    init_guess = np.repeat(1/n,n)
    weight_bounds = ((0.0,1.0),)*n
    constraint1 = {
        'type':'eq',
        'fun': sum_weights_is_1 #Calling the function no need (para1,para2). If you use lambda need to say lambda w: np.sum(w)
    }
    # constraint2 = {
    #     'type':'eq',
    #     'args': (er, target_return,),
    #     'fun': return_is_target
    # }
    # weights_sum_to1 = {
    #     'type':'eq',
    #     'fun': lambda w: np.sum(w) - 1
    # }
    return_is_target = {
        'type' : 'eq',
        'args' : (er,) , #args used in erk.portfolio_return
        'fun' : lambda w, er: erk.portfolio_return(w, er) - target_return
    }
    results = minimize(erk.portfolio_vol,
                       x0=init_guess,
                       args=args,
                       bounds=weight_bounds,
                       #constraints=(constraint1, constraint2),
                       constraints=(constraint1,return_is_target),
                       method='SLSQP')
    weight = results.x
    return weight

print(minimize_vol(0.15,er,cov))