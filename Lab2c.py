from dash import dash
from dash import dcc
from dash import html
from dash.dependencies import Input,Output,State
from dash import dash_table

import plotly.express as px

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from pickle import load
import cvxopt as opt
from cvxopt import solvers
import os

#### Loading Data

file_path = os.path.join(os.getcwd(), 'build_lab2b.sav')
loaded_model = load(open(file_path, 'rb'))
investors = pd.read_csv('InputData.csv', index_col=0)
assets = pd.read_csv('SP500Data.csv', index_col=0)
# add dates
assets.index = pd.to_datetime(assets.index)

########

missing_fractions = assets.isnull().mean().sort_values(ascending=False)

drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))

assets.drop(labels=drop_list, axis=1, inplace=True)
# Fill the missing values with the last value available in the dataset.
assets=assets.ffill()

options = []

for tic in assets.columns:
    #{'label': 'user sees', 'value': 'script sees'}
    mydict = {}
    mydict['label'] = tic #Apple Co. AAPL
    mydict['value'] = tic
    options.append(mydict)


app = dash.Dash(
    __name__,
    external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']
)

app.layout = html.Div([
    html.Div([
        # Add my name here
        html.H3("Junyu Zhang Created", style={
            'textAlign': 'center',
            'color': '#2C3E50',
            'fontSize': '28px',
            'fontWeight': 'bold',
            'marginBottom': '20px',
            'padding': '10px',
            'backgroundColor': '#ECF0F1',
            'borderRadius': '8px'
        }),
        # Dashboard Name
        html.Div([
            html.H3(children='Robo Advisor Dashboard'),
            html.Div([
                html.H5(children='Step 1 : Enter Investor Characteristics '),
            ], style={'display': 'inline-block', 'vertical-align': 'top',
                      'width': '30%',
                      'color': 'black', 'background-color': 'LightGray'}),
            html.Div([
                html.H5(
                    children='Step 2 : Asset Allocation and portfolio performance'),
            ], style={'display': 'inline-block', 'vertical-align': 'top',
                      'color': 'white', 'horizontalAlign': "left",
                      'width': '70%', 'background-color': 'black'}),
        ], style={'font-family': 'calibri'}),

        # All the Investor Characteristics
        # ********************Demographics Features DropDown********
        html.Div([
            html.Div([

                html.Label('Age:', style={'padding': 5}),
                dcc.Slider(
                    id='Age',
                    min=investors['AGE07'].min(),
                    max=70,
                    marks={25: '25', 35: '35', 45: '45', 55: '55', 70: '70'},
                    value=25),
                # Delete Net income here
                html.Label('Income:', style={'padding': 5}),
                dcc.Slider(
                    id='Inccl',
                    min=-1000000,
                    max=3000000,
                    marks={-1000000: '-$1M', 0: '0', 500000: '$500K',
                           1000000: '$1M', 2000000: '$2M', },
                    value=100000),
                
                html.Label('Education Level (scale of 4):',
                           style={'padding': 5}),
                dcc.Slider(
                    id='Edu',
                    min=investors['EDCL07'].min(),
                    max=investors['EDCL07'].max(),
                    marks={1: '1', 2: '2', 3: '3', 4: '4'},
                    value=2),
                
                html.Label('Married:', style={'padding': 5}),
                dcc.Slider(
                    id='Married',
                    min=investors['MARRIED07'].min(),
                    max=investors['MARRIED07'].max(),
                    marks={1: '1', 2: '2'},
                    value=1),
                
                html.Label('Kids:', style={'padding': 5}),
                dcc.Slider(
                    id='Kids',
                    min=investors['KIDS07'].min(),
                    max=investors['KIDS07'].max(),
                    marks=[{'label': j, 'value': j} for j in
                           investors['KIDS07'].unique()],
                    value=3),
                
                # html.Br(),
                html.Label('Occupation:', style={'padding': 5}),
                dcc.Slider(
                    id='Occ',
                    min=investors['OCCAT107'].min(),
                    max=investors['OCCAT107'].max(),
                    marks={1: '1', 2: '2', 3: '3', 4: '4'},
                    value=3),
                
                # html.Br(),
                html.Label('Willingness to take Risk:', style={'padding': 5}),
                dcc.Slider(
                    id='Risk',
                    min=investors['RISK07'].min(),
                    max=investors['RISK07'].max(),
                    marks={1: '1', 2: '2', 3: '3', 4: '4'},
                    value=3),
                
                # html.Br(),
                html.Button(id='investor_char_button',
                            n_clicks=0,
                            children='Calculate Risk Tolerance',
                            style={'fontSize': 14, 'marginLeft': '30px',
                                   'color': 'white',
                                   'horizontal-align': 'left',
                                   'backgroundColor': 'grey'}),
                
                # html.Br(),
            ], style={'width': '80%'}),

        ], style={'width': '30%', 'font-family': 'calibri',
                  'vertical-align': 'top', 'display': 'inline-block'
                  }),

        # ********************Risk Tolerance Charts********
        html.Div([
            html.H5('Step 2 : Select stocks'),
            html.Div([
                html.Div([
                    html.Label('Risk Tolerance (scale of 100) :',
                               style={'padding': 5}),
                    dcc.Input(id='risk-tolerance-text'),

                ], style={'width': '100%', 'font-family': 'calibri',
                          'vertical-align': 'top', 'display': 'inline-block'}),
                
                # Add date selection here
                html.Label('Select date (for viewing asset allocation on that date):', style={'padding': 5}),
                dcc.DatePickerSingle(
                    id='date_picker',
                    min_date_allowed=assets.index.min().date(),
                    max_date_allowed=assets.index.max().date(),
                    initial_visible_month=assets.index.min().date(),
                    date=assets.index.min().date()
                ),
                
                # Add the investment amount
                html.Label('Investment Amount (USD):', style={'padding': 5}),
                    dcc.Input(
                        id='investment_amount',
                        type='number',
                        value=1000000,
                        style={'margin': '5px'}
                    ),
                
                html.Div([
                    html.Label('Select the assets for the portfolio:',
                               style={'padding': 5}),
                    dcc.Dropdown(
                        id='ticker_symbol',
                        options=options,
                        value=['GOOGL', 'FB', 'GS', 'MS', 'GE', 'MSFT'],
                        multi=True
                        # style={'fontSize': 24, 'width': 75}
                    ),
                    html.Button(id='submit-asset_alloc_button',
                                n_clicks=0,
                                children='Submit',
                                style={'fontSize': 12, 'marginLeft': '25px',
                                       'color': 'white',
                                       'backgroundColor': 'grey'}

                                ),
                ], style={'width': '100%', 'font-family': 'calibri',
                          'vertical-align': 'top', 'display': 'inline-block'}),
            ], style={'width': '100%', 'display': 'inline-block',
                      'font-family': 'calibri', 'vertical-align': 'top'}),

            html.Div([
                html.Div([
                    html.H4("Asset Allocation", style={'textAlign': 'center', 'marginBottom': '10px'}),
                    dcc.Graph(id='Asset-Allocation'),
                    # Add the position table
                    html.H4("Position Table", style={'textAlign': 'center', 'marginTop': '20px'}),
                    dash_table.DataTable(
                        id='position_table',
                        columns=[
                            {'name': 'Asset', 'id': 'Asset'},
                            {'name': 'Shares', 'id': 'Shares'}
                        ],
                        data=[],
                        style_table={'marginTop': '20px', 'width': '90%', 'margin': 'auto'},
                        style_cell={
                            'textAlign': 'center',
                            'padding': '5px',
                            'fontFamily': 'calibri',
                            'fontSize': '14px'
                        },
                        style_header={
                            'backgroundColor': '#2C3E50',
                            'fontWeight': 'bold',
                            'color': 'white'
                        },
                        style_data={
                            'backgroundColor': '#ECF0F1',
                            'color': 'black'
                        }
                    )
                ], style={'width': '50%', 'vertical-align': 'top',
                          'display': 'inline-block',
                          'font-family': 'calibri',
                          'horizontal-align': 'right'}),
                html.Div([
                    html.H4("Portfolio Performance Over Time", style={'textAlign': 'center', 'marginBottom': '10px'}),
                    # Add the pie chart
                    dcc.Graph(id='Performance'),
                    dcc.Graph(id='pie-chart')
                ], style={'width': '50%', 'vertical-align': 'top',
                          'display': 'inline-block',
                          'font-family': 'calibri',
                          'horizontal-align': 'right'}),
            ], style={'width': '100%', 'vertical-align': 'top',
                      'display': 'inline-block',
                      'font-family': 'calibri', 'horizontal-align': 'right'}),

        ], style={'width': '70%', 'display': 'inline-block',
                  'font-family': 'calibri', 'vertical-align': 'top',
                  'horizontal-align': 'right'}),
    ], style={'width': '70%', 'display': 'inline-block',
              'font-family': 'calibri', 'vertical-align': 'top'}),

])


# Asset allocation given the Return, variance
def get_asset_allocation(riskTolerance, stock_ticker):

    assets_selected = assets.loc[:, stock_ticker]
    return_vec = np.array(assets_selected.pct_change().dropna(axis=0)).T
    n = len(return_vec)
    mus = 1 - riskTolerance

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(return_vec))
    pbar = opt.matrix(np.mean(return_vec, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = solvers.qp(mus * S, -pbar, G, h, A, b)
    w = portfolios['x'].T
    Alloc = pd.DataFrame(
        data=np.array(portfolios['x']),
        index=assets_selected.columns
    )

    # Calculate efficient frontier weights using quadratic programming
    returns_final = (np.array(assets_selected) * np.array(w))
    returns_sum = np.sum(returns_final, axis=1)
    returns_sum_pd = pd.DataFrame(returns_sum, index=assets.index)
    returns_sum_pd = returns_sum_pd - returns_sum_pd.iloc[0, :] + 100
    return Alloc, returns_sum_pd


# Callback for the graph
# This function takes all the inputs and computes the cluster and the risk tolerance
@app.callback(
    [
        Output('risk-tolerance-text', 'value')
    ],
    [
        Input('investor_char_button', 'n_clicks')
    ],
    [
        State('Age', 'value'),
        State('Inccl', 'value'),
        State('Risk', 'value'),
        State('Edu', 'value'),
        State('Married', 'value'),
        State('Kids', 'value'),
        State('Occ', 'value')
    ],
    prevent_initial_call=True,
)
def update_risk_tolerance(
        n_clicks, Age, Inccl, Risk, Edu, Married, Kids, Occ
):
    X_input = [[Age, Edu, Married, Kids, Occ, Inccl, Risk]]
    RiskTolerance = loaded_model.predict(X_input)

    # Using linear regression to get the risk tolerance within the cluster.
    return list([round(float(RiskTolerance * 100), 2)])


@app.callback(
    [
        Output('Asset-Allocation', 'figure'),
        Output('Performance', 'figure')
    ],
    [
        Input('submit-asset_alloc_button', 'n_clicks'),
        Input('risk-tolerance-text', 'value')
    ],
    [
        State('ticker_symbol', 'value')
    ],
    prevent_initial_call=True
)

def update_asset_allocationChart(n_clicks, risk_tolerance, stock_ticker):

    Allocated, InvestmentReturn = get_asset_allocation(risk_tolerance,
                                                       stock_ticker)

    return [{'data': [go.Bar(
        x=Allocated.index,
        y=Allocated.iloc[:, 0],
        marker=dict(color='red'),
    ),
    ],
        'layout': {'title': " Asset allocation - Mean-Variance Allocation"}

    },
        {'data': [go.Scatter(
            x=InvestmentReturn.index,
            y=InvestmentReturn.iloc[:, 0],
            name='OEE (%)',
            marker=dict(color='red'),
        ),
        ],
            'layout': {'title': "Portfolio value of $100 investment"}

        }]

# Call back for position_table
@app.callback(
    Output('position_table', 'data'),
    [Input('Asset-Allocation', 'figure'),
     Input('investment_amount', 'value')]
)

def update_positions(asset_alloc_fig, investment_amount):
    if not asset_alloc_fig or not investment_amount:
        return dash.no_update
    # Get data from bar chart
    bar_data = asset_alloc_fig.get('data', [])[0]
    symbols = bar_data.get('x', [])
    weights = bar_data.get('y', [])
    # Use initial price of that asset
    starting_prices = assets.iloc[0]
    position_data = []
    # Calculate the weights
    for sym, weight in zip(symbols, weights):
        price = starting_prices[sym]
        shares = round((investment_amount * weight) / price)
        position_data.append({'Asset': sym, 'Shares': shares})
    return position_data

# Call back for pie chart
@app.callback(
    Output('pie-chart', 'figure'),
    [Input('date_picker', 'date'),
     Input('position_table', 'data')]
)
def update_pie_chart(selected_date, position_data):
    if not position_data or not selected_date:
        return dash.no_update
    # Convert the selected date to a timestamp
    selected_date = pd.to_datetime(selected_date)
    try:
        # Get the price of asset for that day
        prices_on_date = assets.loc[selected_date]
    except KeyError:
        # If not, return an empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for the selected date",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Portfolio Allocation on Selected Date",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return fig
    values = []
    labels = []
    for row in position_data:
        sym = row['Asset']
        shares = row['Shares']
        price = prices_on_date[sym]
        value = shares * price
        values.append(value)
        labels.append(sym)
    # Add colors    
    colors = px.colors.qualitative.Plotly
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        marker=dict(colors=colors)
    )])
    # Enable the pie chart calculate percentage display
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_layout(
        title={
            'text': "Portfolio Allocation on Selected Date",
            'x': 0.5,
            'xanchor': 'center'
        },
        title_font=dict(
            family="Calibri",
            size=24,
        )
    )
    return fig

if __name__ == '__main__':
    app.run_server()
