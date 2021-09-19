import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import plotly.express as px
import plotly.graph_objects as go


def process_data():
    """
    This Function processes data from its raw form.
    """
    data = pd.read_csv('final_test.csv')
    data.drop('Unnamed: 0', axis=1, inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    return data


def getTimeSeriesData(data):
    data = data[['Date', 'Price']]
    data.set_index('Date', inplace=True)
    data = data.sort_index()
    return data


def px_graph(df, timeframe=''):
    """
        parameters:
            df = dataframe with index as datetime
            timeframe as W,M,AS.
    """
    if timeframe == '':
        pass
    else:
        df = df.resample(timeframe).sum()
    scatter_data = go.Scatter(x=df.index,
                              y=df.Price)
    layout = go.Layout(title='Revenue Graphs ({})'.format(timeframe), xaxis=dict(title='Date'), yaxis=dict(title='Revenue'))
    fig = go.Figure(data=[scatter_data], layout=layout)
    fig.show()


data = process_data()
df = getTimeSeriesData(data)
px_graph(df, 'Q')
