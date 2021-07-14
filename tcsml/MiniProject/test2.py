import dash
import pandas as pd
import plotly.express as px

data = pd.read_csv('Dataset_proj1.csv')
data.drop('Unnamed: 0', axis=1, inplace=True)
#print(data.head())

#print(data["Item"].unique())

# fig = px.pie(data = data, names = "Item")
data['Date'] = pd.to_datetime(data['Date'])
import datetime
data1 = data[data['Date']>= datetime.datetime(2019,7,5)]
data2 = data[data["Date"]<=datetime.datetime(2019,7,5)+ datetime.timedelta(days = 84)]

int_df = pd.merge(data1, data2, how ='inner', on =['Date','Time','Customer_ID','Item'])

fig = px.histogram(int_df, x='Item', barmode='group')
fig.show()
