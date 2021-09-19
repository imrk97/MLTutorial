# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 20:53:19 2021

@author: rohan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
def process_data():
    '''
    This Function processes data from its raw form.
    '''
    data = pd.read_csv('final_test.csv')
    data.drop('Unnamed: 0', axis=1, inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    return data
def pop_item(data, year, month, day):
    '''
        parameters: data is the dataframe,
                    year, month, day represents a date
                    days_report is the number of days you want the report on.
        
    '''
    '''data1 = data[data['Date']>= datetime.datetime(year,month,day)]
    data2 = data[data["Date"]<=datetime.datetime(year,month,day)+ datetime.timedelta(days = days_report)]'''
    int_df = data[data['Date']==datetime.datetime(year, month, day)]
    print(int_df)
    #sns.countplot(x='Item', data = int_df, order = int_df['Item'].value_counts().iloc[:6].index)
    return str(int_df['Item'].value_counts().iloc[:1].index[0])

data = process_data()
popular_item = pop_item(data,2021,7,20)
