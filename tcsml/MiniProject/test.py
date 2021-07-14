import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

def add_revenue(item):
    return items[item[0]]

data = pd.read_csv('Dataset_proj1.csv')

data.drop('Unnamed: 0',axis = 1, inplace=True)

data['Date'] = pd.to_datetime(data['Date'])
data.Item.unique()
items = {'Palak Paneer':80, 'Palak Aloo':50, 'Roti': 30, 'Dal Tadka': 60, 'Shabnam Curry': 50,
       'Garlic Naan':40, 'Naan':30, 'Chana Paneer':60, 'Achari Paneer':80,
       'Aloo Gobi':50, 'Paneer Makhani':80, 'Paneer Butter Masala':100,
       'Dal Makhani':50, 'Chana Masala':50, 'Vegetable Green Masala':50,
       'Bhindi Masala':60, 'Malai Kofta':50, 'Paneer Shahi Korma': 100, 'Aloo Matar':50,
       'Matar Paneer':80, 'Paratha':40, 'Mixed Vegetable Curry':60,
       'Baigan Bharta':80, 'Navratan Korma':60, 'Mixed Vegetable Vindaloo': 70}
data['Price'] = data[['Item']].apply(add_revenue, axis = 1)
data.to_csv('data.csv')
sns.countplot(x='Item',data = data,order = data[['Item']].value_counts().index)
sns.countplot(x='Item', data = int_df, order = int_df['Item'].value_counts().iloc[:3].index)
data['Item'].nunique()

import datetime
data1 = data[data['Date']>= datetime.datetime(2019,7,5)]
data2 = data[data["Date"]<=datetime.datetime(2019,7,5)+ datetime.timedelta(days = 84)]

int_df = pd.merge(data1, data2, how ='inner', on =['Date','Time','Customer_ID','Item'])

def impute_price(cols):
    item = cols
    return items[item]