#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import plotly.express as px
import plotly.graph_objects as go
def process_data():
    '''
    This Function processes data from its raw form.
    '''
    data = pd.read_csv('final_test.csv')
    data.drop('Unnamed: 0',axis = 1, inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    return data
def getTimeSeriesData(data):
    data = data[['Date','Price']]
    data.set_index('Date', inplace = True)
    return data
data = process_data()
data = getTimeSeriesData(data)

#df_month.plotot()
fig = px.bar(df_month)
#fig = px.area(data,  facet_col_wrap=2)

fig.show()


# In[23]:


df_month.Price.diff().plot()


# In[32]:


data=data.sort_index()


# In[37]:


scatter_data = go.Scatter(x=data.index,
                         y=data.Price)
layout = go.Layout(title='Revenue Graphs', xaxis=dict(title='Date'), yaxis=dict(title='Revenue'))
fig = go.Figure(data=[scatter_data], layout=layout)
fig.show()


# In[41]:


scatter_data_month = go.Scatter(x=df_month.index,
                         y=df_month.Price)
layout = go.Layout(title='Revenue Graphs Monthly' , xaxis=dict(title='Date'), yaxis=dict(title='Revenue'))
fig = go.Figure(data=[scatter_data_month], layout=layout)
fig.show()


# In[39]:


df_month = data.resample("M").mean()
df_week = data.resample("W").mean()


# In[42]:


scatter_data_weekly = go.Scatter(x=df_week.index,
                         y=df_week.Price)
layout = go.Layout(title='Revenue Graphs Weekly', xaxis=dict(title='Date'), yaxis=dict(title='Revenue'))
fig = go.Figure(data=[scatter_data_weekly], layout=layout)
fig.show()


# In[ ]:




