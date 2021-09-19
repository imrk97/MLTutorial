import dash
import dash_core_components as dcc
import dash_html_components as html
import dcc as dcc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import plotly.express as px
import plotly.graph_objects as go
import datetime

data_sale = pd.read_csv("Dataset_proj1.csv")
data_menu = pd.read_csv("Menu.csv")

app = dash.Dash(_name_)

app.layout = html.Div(
    children=[
        html.H1(children="Daily Statistics",
                style={"fontSize": "48px", "color": "red"}, ),
        html.P(
            children="Sales Graphical Analysis",
        ),
        dcc.DatePickerSingle(
            id='my-date-picker-single',
            min_date_allowed=date(2017, 1, 1),
            initial_visible_month=date(2017, 1, 1),
            date=date(,,)
),
html.Div(id='output-container-date-picker-single'),
dcc.Graph(id='graph-with-date'),

]
)

@app.callback(
    Output('output-container-date-picker-single', 'children', 'graph-with-date', 'figure'),
    Input('my-date-picker-single', 'date'))
def update_output(date_value):
    # string_prefix = 'You have selected: '
    if date_value is not None:
        df1 = data_sale.loc[date_value]
        for i in df1['Item']:
            sum = data_menu['Price'].loc[df1['Item'] == i]

    result = "Total sales today = " + str(sum)
    return result, fig


if _name_ == "_main_":
    app.run_server(debug=True)