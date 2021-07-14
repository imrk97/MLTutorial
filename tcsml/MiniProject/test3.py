import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
def process_data():
    '''
    This Function processes data from its raw form.
    '''
    data = pd.read_csv('Dataset_proj1.csv')
    data.drop('Unnamed: 0',axis = 1, inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    return data
def pop_item(data, year, month, day, days_report):
    data1 = data[data['Date']>= datetime.datetime(year,month,day)]
    data2 = data[data["Date"]<=datetime.datetime(year,month,day)+ datetime.timedelta(days = days_report)]
    int_df = pd.merge(data1, data2, how ='inner', on =['Date','Time','Customer_ID','Item'])
    sns.countplot(x='Item', data = int_df, order = int_df['Item'].value_counts().iloc[:3].index)
    return str(int_df['Item'].value_counts().iloc[:1].index[0])
#data = pd.read_csv('Dataset_proj1.csv')
data = process_data()
popular_item = pop_item(data,2019,8,7,90)