# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 17:57:53 2021

@author: rohan
"""

import dash
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

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
    #data.set_index('Date', inplace = True)
    return data

df = getTimeSeriesData(process_data())

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id="ticker",
        options=[{"label": x, "value": x} 
                 for x in df.columns[1:]],
        value=df.columns[1],
        clearable=False,
    ),
    dcc.Graph(id="time-series-chart"),
])

@app.callback(
    Output("time-series-chart", "figure"), 
    [Input("ticker", "value")])
def display_time_series(ticker):
    fig = px.line(df, x='Date', y=ticker)
    return fig

app.run_server(debug=True)