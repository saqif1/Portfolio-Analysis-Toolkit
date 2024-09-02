import numpy as np
import pandas as pd
import scipy.stats
import scipy.stats.stats as st
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import jarque_bera
import yfinance
import yfinance as yf
import matplotlib.pyplot as plt

def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    #Deduct the risk_free from our data, which means we need to follow data freq NOT assume annual
    excess_ret = r - rf_per_period
    #Now we use it to gete annualised returns
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def annualised_return(ser: pd.Series, frequency='m'):
    '''
    :param: Series
    :return: Annualised return in %
    '''
    num_periods = ser.shape[0]

    if frequency=='m':
        num_periods_in_a_year = 12
    elif frequency=='y':
        num_periods_in_a_year = 1
    elif frequency=='d':
        num_periods_in_a_year = 252
    else:
        TypeError('Choose the correct frequency of observation.')

    compounded_return = (ser + 1).prod()

    annualised_return =  (compounded_return ** (num_periods_in_a_year / num_periods)) - 1

    return annualised_return

def annualised_volatility(ser: pd.Series, frequency='m'):
    '''
    :param: Series
    :return: Annualised volatility
    '''
    if frequency=='m':
        num_periods_in_a_year = 12
    elif frequency=='y':
        num_periods_in_a_year = 1
    elif frequency=='d':
        num_periods_in_a_year = 252
    else:
        TypeError('Choose the correct frequency of observation.')

    std_dev = ser.std()

    annualised_volatility = std_dev*np.sqrt(num_periods_in_a_year)

    return annualised_volatility


def drawdowns(Returns: pd.Series):
    '''
    :param Returns:
    :return: Series containing Drawdowns
    '''
    wealth_index = 1000*((Returns+1).cumprod())
    previous_peaks = wealth_index.cummax()

    drawdowns = (wealth_index - previous_peaks)/previous_peaks

    df = drawdowns
    df.index = Returns.index

    return df

def semideviation(Ret: pd.Series):
    '''
    :param Ret:
    :return: Returns semideviation for Returns that are <0
    '''
    negative_ret = Ret[Ret<0]
    semideviation = negative_ret.std(ddof=0)

    return semideviation

# def skewness1(ret: pd.Series):
#     '''
#     :param ret:
#     :return: Kurtosis with Vijay's definition of non-excess and bias as True
#     '''
#     kurt = st.skew(ret, bias=True)
#     return kurt

def skewness(ret: pd.Series):
    '''
    :param ret:
    :return: Kurtosis with Vijay's definition of non-excess and bias as True
    '''
    demeaned_r = ret - ret.mean()
    exp = demeaned_r**3
    std_dev = ret.std(ddof=0)
    return exp.mean()/std_dev**3

def kurtosis(ret: pd.Series):
    '''
    :param ret:
    :return: Kurtosis with Vijay's definition of non-excess and bias as True
    '''
    kurt = st.kurtosis(ret, bias=True, fisher=False)
    return kurt

def historicalVaR(ret: pd.Series, level=0.05):
    '''
    :param ret: Return series
    :param level: Significance level of left-tail only
    :return: historical var at specified level of alpha
    '''
    historicalVaR = -ret.quantile(q=level)

    return historicalVaR

def conditionalVaR(ret: pd.Series, level=0.05):
    upper_limit = ret.quantile(q=level)
    conditionalVaR = -ret[ret<upper_limit].mean()

    return conditionalVaR

def parametric_gaussianVaR(ret: pd.Series, level=0.05):
    '''
    :param ret: Return series
    :param level: Significance level of left-tail only
    :return: Parametric Gaussian var at specified level of alpha
    '''
    parametric_gaussianVaR = -norm.ppf(loc=ret.mean(), scale=ret.std(ddof=0), q=level)
    return parametric_gaussianVaR


def cornish_fisher_modifiedVaR(ret: pd.Series, level = 0.05):
    s = skewness(ret)
    k = kurtosis(ret)

    z = norm.ppf(q=level)

    z = (z +
         (z ** 2 - 1) * s / 6 +
         (z ** 3 - 3 * z) * (k - 3) / 24 -
         (2 * z ** 3 - 5 * z) * (s ** 2) / 36
         )
    return -( (ret.mean() + z*ret.std(ddof=0)) )

def is_normal(ret, level=0.05):
    '''
    Uses Jarque-Bera test of normality
    :param ret:
    :return: Returns True if dis is normal
    '''
    jb_value, p_value = scipy.stats.jarque_bera(ret)

    return p_value > level

def portfolio_return(weights, returns):
    '''
    Weights -> Returns
    :param weights: weights in tuple datatype
    :param returns:
    :return:
    '''
    return weights.T @ returns

def portfolio_vol(weights, cov):
    '''
    Weights -> Portfolio weighted volatility
    :param weights:
    :param cov:
    :return:
    '''
    return (weights.T @ cov @ weights)**0.5

def ef_2asset(n_points, er, cov):
    '''
    n_points, er, cov -> Return, Volatility in df format
    :param n_points:
    :param er:
    :param cov:
    :return:
    '''
    if er.shape[0] != 2:
        raise ValueError('2asset_ef can only accept 2-asset portfolios')
    weights = [np.array([w,1-w]) for w in np.linspace(0,1,num=n_points)]
    rets = [portfolio_return(w,er) for w in weights]
    vol = [portfolio_vol(w,cov) for w in weights]

    ef = pd.DataFrame({'R':rets,
                       'Vol':vol})
    return ef

def minimize_vol(target_return, er, cov):
    '''
    target_return, er, cov -> a set of weight
    :param target_return:
    :param er:
    :param cov:
    :return:
    '''
    n_assets = er.shape[0]
    args = (cov, )  #args that is used in erk.portfolio_vol() function that we wanna reduce
    init_guess = np.repeat(1/n_assets, n_assets) #initial guess
    bounds = ((0.0, 1.0), )*n_assets #bounds repeated for each weight combination
    weights_sum_to1 = {
        'type':'eq',
        'fun': lambda w: np.sum(w) - 1
    }
    return_is_target = {
        'type' : 'eq',
        'args' : (er,) , #args used in erk.portfolio_return
        'fun' : lambda w, er: portfolio_return(w, er) - target_return
    }
    result = minimize(portfolio_vol, init_guess,
                 args=args,
                 bounds=bounds,
                 constraints=(weights_sum_to1, return_is_target),
                 method= 'SLSQP',
                 options={'disp': False}
                 )
    w = result.x
    return w