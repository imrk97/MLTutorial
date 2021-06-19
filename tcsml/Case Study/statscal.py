# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 01:10:03 2021

@author: rohan
"""
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

def calc_vif(X):

    # Calculating VIF
    '''
    This function will return a dataframe that will contain the vif values of each of the features
    '''
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)
def cal_corr_pval(df, series):
    corr_df = pd.DataFrame(columns=['r','p'])
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            r, p = stats.pearsonr(series, df[col])
            corr_df.loc[col] = [r, p]
    return corr_df