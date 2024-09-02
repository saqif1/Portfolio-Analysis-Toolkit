import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import edhec_risk_kit as erk

#1. Read dataset
# Read the num_firms dataset
ind_nfirms = pd.read_csv('data/ind30_m_nfirms.csv', index_col=0)
ind_nfirms.index = pd.to_datetime(ind_nfirms.index, format='%Y%m').to_period('M')
ind_nfirms.columns = ind_nfirms.columns.str.strip()
#Read the mean_firm_size dataset
ind_size = pd.read_csv('data/ind30_m_size.csv', index_col=0)
ind_size.index = pd.to_datetime(ind_size.index, format='%Y%m').to_period('M')
ind_size.columns = ind_size.columns.str.strip()

#2. Create market cap df by multiplying nfirms and mean_size
ind_mkt = ind_nfirms * ind_size
#print(ind_mkt)

#3. Total market cap for that time
ind_tot_mkt = ind_mkt.sum(axis='columns')
#print(ind_tot_mkt)

#4. Obtain fraction of ind over tot_mkt_cap as its weight for that time period
ind_capweight = ind_mkt.divide(ind_tot_mkt, axis='rows') #divide according to index values or row-wise as that is the date
print(ind_capweight['1926'].sum(axis='columns'))