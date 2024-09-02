import numpy as np
import pandas as pd
import edhec_risk_kit as erk
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#Max Sharpe Ratio Portfolio

# 1. Load data and define ind df
ind = pd.read_csv('data/ind30_m_vw_rets.csv', index_col=0)/100
ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
ind.columns = ind.columns.str.strip() #Strips whitespace from headers

# 2. Define er to be 1996:2000. er is also annualised returns
er = erk.annualize_rets(ind['1996':'2000'], periods_per_year=12)

# 3. Define cov matrix
cov = ind['1996':'2000'].cov()

# 4. Helper Functions
def portfolio_return(w, er):
    return w.T @ er

def portfolio_volatility(w, cov): #remember to sqrt
    return (w.T @ cov @ w)**0.5

# 5. Create efficient frontier. Given set of weights, numerically achieve target return given that has minimum volatility

def minimize_vol(target_er, er, cov):
    def sum_w_is_1(w):
        return np.sum(w) - 1

    def return_is_target(w, er, target_er):
        return portfolio_return(w, er) - target_er

    target_er = target_er

    n_weights = er.shape[0]  # get number of assets or weights
    bounds = ((0.0, 1.0),) * n_weights  # each w must >0 and <1
    init_guess = np.repeat(1 / n_weights, n_weights)  # guess input with each w being 1/n or equally weighted
    constraint1 = {
        'type': 'eq',
        'fun': sum_w_is_1
    }
    constraint2 = {
        'type': 'eq',
        'args': (er, target_er,),
        'fun': return_is_target
    }
    args = (cov,)
    results = minimize(portfolio_volatility,
                       x0=init_guess, args=args,
                       bounds=bounds,
                       constraints=(constraint1, constraint2),
                       method='SLSQP'
                       )
    return results.x

def optimum_weights(n_points, er, cov):
    n_points = n_points
    target_returns = np.linspace(er.min(), er.max(), num=n_points)
    opt_weights = [minimize_vol(target_return, er, cov) for target_return in target_returns]
    return opt_weights

# 6. Plot efficient frontier
weights = optimum_weights(20, er, cov)
return_y = [portfolio_return(w, er) for w in weights]
volatility_x = [portfolio_volatility(w, cov) for w in weights]

ret_vol_df = pd.DataFrame({
    'returns':return_y,
    'volatility':volatility_x
})
ax = ret_vol_df.plot.line(x='volatility', y='returns', marker='.', title='Portfolio Mean VS. Variance')
ax.set_xlim(left=0)
#plt.show()

# 7. Find MSR using minimizer to minimize negative sharpe ratio (or maximise the sharpe ratio)
def neg_sharpe_ratio(w, er, cov, rf):
    ann_rets = portfolio_return(w,er)
    ann_vol = portfolio_volatility(w,cov) #using cov hence no need to *np.sqrt(T)
    sr = (ann_rets-rf)/ann_vol
    return -sr #we want to maximise sharpe therefore minimise negative sharpe

def msr(er, cov, rf):
    '''
    Returns weights of the Max. Sharpe Ratio portfolio given expected return, cov, risk-free rate
    '''
    def w_sum_to_1(w):
        return np.sum(w) - 1

    n = er.shape[0]
    args = (er, cov, rf, )
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    constraint1 = {
        'type':'eq',
        'fun': w_sum_to_1
    }
    results = minimize(neg_sharpe_ratio, x0=init_guess, args=args, bounds=bounds, constraints=constraint1,method='SLSQP')
    return results.x

rf = 0.1
w_msr = msr(er,cov,rf)
er_msr = portfolio_return(w_msr, er)
vol_msr = portfolio_volatility(w_msr, cov)
#ax.plot(vol_msr, er_msr, marker='X', color='purple')
cml_x = [0, vol_msr]
cml_y = [rf, er_msr]
# cml_m = (cml_y[1]-cml_y[0])/(cml_x[1]-cml_x[0])
# cml_y_intercept = rf
# def cml(x, gradient=cml_m, y_intercept=cml_y_intercept):
#     y = gradient*x + y_intercept
#     return y
# cml_x_list = np.linspace(0,0.08,10)
# cml_y_list = [cml(x,gradient=cml_m,y_intercept=cml_y_intercept) for x in cml_x_list]
# ax.plot(cml_x_list, cml_y_list, color='green', marker='o', linestyle='dashed')
ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed')
#plt.show()

# 8. Display the point of equally weighted portfolio's expected return (y) and volatitlity (x) - NO er required
# Naive method: Just take equal-weights
n = er.shape[0]
w_ew = np.repeat(1/n,n)
er_ew = portfolio_return(w_ew, er)
vol_ew = portfolio_volatility(w_ew, cov)
ax.plot([vol_ew], [er_ew], color='goldenrod', marker='^', markersize=10)
#plt.show()

# 9. Display point of Global Minimum Volatility' portfolio expected return (y) and volatitlity (x) - NO er required
#Note: GMV relies only on an estimate of cov (volatility)
# Estimating cov is easier than estimating expected return.
def gmv(cov):
    '''
    Returns weight of GMV portfolio given cov matrix.
    '''
    n = er.shape[0]
    # when all returns are same, minimiser finds w that reduces the cov(min vol) as it cannot do anything with er.
    return msr(er=np.repeat(1,n), cov=cov, rf=rf)
w_gmv = gmv(cov)
er_gmv = portfolio_return(w_gmv, er)
vol_gmv = portfolio_volatility(w_gmv, cov)
ax.plot([vol_gmv],[er_gmv], color='midnightblue', marker='X', markersize=10)
plt.show()

#Summary: MSR portfolio is best but its reliance on er(hard to estimate) makes it swing crazily (non-robust) hence, we look at
# 2 other portfolios: Equally-weighted portfolio(naive) and Global Minimum Volatility Portfolio (requires cov).