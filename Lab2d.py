from dash import dash, dash_table, dcc, html
from dash.dependencies import Input,Output,State
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from pickle import load
import cvxopt as opt
from cvxopt import solvers
import os
import plotly.express as px
import datetime


#### Loading Data
file_path = os.path.join(os.getcwd(), 'build_lab2b.sav')
loaded_model = load(open(file_path, 'rb'))

try:
    investors = pd.read_csv(
        './build labs data/InputData.csv',
        index_col=0
    )
    assets = pd.read_csv(
        './build labs data/SP500Data.csv',
        index_col=0
    )
except FileNotFoundError:
    investors = pd.read_csv('InputData.csv', index_col=0)
    assets = pd.read_csv('SP500Data.csv', index_col=0)
########

missing_fractions = assets.isnull().mean().sort_values(ascending=False)

drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))

assets.drop(labels=drop_list, axis=1, inplace=True)
assets.index = pd.to_datetime(assets.index)
# Fill the missing values with the last value available in the dataset.
assets=assets.ffill()
# Set several key dates as labels for display
first_date = pd.to_datetime(assets.index.min()).date()
last_date = pd.to_datetime(assets.index.max()).date()
dates = pd.to_datetime(assets.index)
mid_date = dates[len(dates) // 2].date()

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
    html.H2('Junyu Zhang'),
    html.Div([
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
                    html.Label(
                        'Starting Capital (USD):',
                        style={'padding': 5}
                    ),
                    dcc.Input(
                        id='starting-capital',
                        type='number',
                        value=1000000
                    ),
                    html.Label('Risk Tolerance (scale of 100) :',
                               style={'padding': 5}),
                    dcc.Input(id='risk-tolerance-text'),

                ], style={'width': '100%', 'font-family': 'calibri',
                          'vertical-align': 'top', 'display': 'inline-block'}),

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
                    dcc.Graph(id='Asset-Allocation'),
                    dash_table.DataTable(id='starting-positions')
                ], style={'width': '50%', 'vertical-align': 'top',
                          'display': 'inline-block',
                          'font-family': 'calibri',
                          'horizontal-align': 'right'}),
                html.Div([
                    dcc.Graph(id='Performance'),
                    dcc.Slider(
                        id='date-slider',
                        min=first_date.toordinal(),
                        max=last_date.toordinal(),
                        step=1,
                        # Set several key dates as labels for display
                        marks={
                            first_date.toordinal(): first_date.strftime("%Y-%m-%d"),
                            mid_date.toordinal(): mid_date.strftime("%Y-%m-%d"),
                            last_date.toordinal(): last_date.strftime("%Y-%m-%d")
                        },
                        value=first_date.toordinal(),
                        # Task2: Add the update mode drag to facilitate real-time updates
                        updatemode='drag',
                        tooltip={
                            "placement": "bottom",
                            "always_visible": False
                        }
                    ),
                    dcc.Graph(id='weights-pie')
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
                # ---- Task3: Add a scatter plot with trendline for alpha and beta (call back function is listed on the buttom) ----
                html.Div([
                    dcc.Graph(
                        id='alpha-beta',
                        style={
                            'width': '90%',
                            'height': '800px'
                        }
                    )
                ],
                style={
                    'maxWidth': '1600px',  
                    'margin': '20px auto',
                    'textAlign': 'center'
                })
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
        Input(
            'submit-asset_alloc_button',
            'n_clicks'
        ),
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


@app.callback(
    Output(
        'starting-positions', 'data'
    ),
    State('starting-capital', 'value'),
    Input('Asset-Allocation', 'figure'),
    prevent_initial_call=True
)
def calc_starting_positions(starting_capital, alloc_dta):
    cash_position = np.array(alloc_dta['data'][0]['y']) * starting_capital
    prices = np.array(assets.loc['2018-01-02', alloc_dta['data'][0]['x']])
    shares = [round(x) for x in cash_position / prices]
    return pd.DataFrame({
        'Asset': alloc_dta['data'][0]['x'],
        'position': shares
    }).to_dict('records')


@app.callback(
    Output('weights-pie', 'figure'),
    [
        Input('date-slider', 'value'),
        Input('starting-positions', 'data')
    ]
)

def update_pie_chart(date_value, starting_positions):
    try:
        starting_positions = pd.DataFrame.from_dict(starting_positions)
        # Converts the selected date to string format
        date_str = datetime.date.fromordinal(date_value).strftime("%Y-%m-%d")
        # Task1: If the selected date is not in the assets index, replace by the nearest valid date
        if date_str not in assets.index:
            valid_dates = pd.to_datetime(assets.index)
            chosen_date = pd.to_datetime(date_str)
            closest_date = valid_dates[(valid_dates - chosen_date).abs().argsort()[0]]
            date_str = closest_date.strftime("%Y-%m-%d")
        starting_positions['prices'] = assets.loc[
            date_str,
            starting_positions['Asset'].values
        ].values
        starting_positions['value'] = starting_positions['prices'] * starting_positions['position']
        fig = px.pie(
            starting_positions,
            values='value',
            names='Asset',
            title = 'Portfolio weights as of ' + date_str
        )
    except KeyError:
        fig = px.pie()

    return fig

from sklearn.metrics import r2_score
import yfinance as yf

@app.callback(
    Output('alpha-beta', 'figure'),
    [
        Input('submit-asset_alloc_button', 'n_clicks'),
        Input('risk-tolerance-text', 'value')
    ],
    [
        State('ticker_symbol', 'value')
    ],
    prevent_initial_call=True
)

# Here I use the daily log return method because the resulting graph is more compact and beautiful. 
# If you want to use the method taught in class, I also add it below and you can see my comments below.
def update_regression_scatter(n_clicks, risk_tolerance, stock_ticker):
    """
    Calculate the linear regression between the overall portfolio and SP500 daily log returns,
    and plot the scatter with the regression line.
    """
    # 1) Get the portfolio value series
    Allocated, InvestmentReturn = get_asset_allocation(risk_tolerance, stock_ticker)
    # InvestmentReturn is a cumulative value series starting from 100
    # Select the first column
    portfolio_val = InvestmentReturn.iloc[:, 0].dropna()

    # 2) Calculate the portfolio logarithmic returns
    portfolio_log_returns = np.log(portfolio_val / portfolio_val.shift(1)).dropna()

    # 3) Read the SP500 benchmark data
    try:
        benchmark_df = pd.read_csv('SP500Index.csv', index_col=0)
        benchmark_df.index = pd.to_datetime(benchmark_df.index)
        benchmark_df = benchmark_df.sort_index()
        sp500_price = benchmark_df.iloc[:, 0].dropna()
    except FileNotFoundError:
        # Download SP500 data from Yahoo Finance
        start_date = portfolio_log_returns.index.min().strftime('%Y-%m-%d')
        end_date = portfolio_log_returns.index.max().strftime('%Y-%m-%d')
        benchmark_df = yf.download('^GSPC', start=start_date, end=end_date)
        if 'Close' in benchmark_df.columns:
            sp500_price = benchmark_df['Close'].dropna()
        else:
            raise KeyError("'Close' not found in downloaded benchmark data")
    benchmark_df.index = pd.to_datetime(benchmark_df.index)
    benchmark_df = benchmark_df.sort_index()
    sp500_price = benchmark_df['Close'].dropna()
    # Calculate the SP500 logarithmic returns
    sp500_log_returns = np.log(sp500_price / sp500_price.shift(1)).dropna()

    # 4) Align the data and perform regression
    combined = pd.concat([portfolio_log_returns, sp500_log_returns], axis=1, join='inner')
    combined.columns = ['portfolio', 'benchmark']
    
    if len(combined) < 2:
        # Not enough data to perform regression
        return go.Figure()

    # Regression: portfolio = intercept + slope * benchmark
    slope, intercept = np.polyfit(combined['benchmark'], combined['portfolio'], 1)
    # Calculate R^2
    predicted = slope * combined['benchmark'] + intercept
    r2 = r2_score(combined['portfolio'], predicted)

    # 5) Prepare scatter and regression line traces
    scatter_trace = go.Scatter(
        x=combined['benchmark'],
        y=combined['portfolio'],
        mode='markers',
        name='Daily Returns'
    )
    # Draw the regression line
    x_min, x_max = combined['benchmark'].min(), combined['benchmark'].max()
    line_x = np.linspace(x_min, x_max, 50)
    line_y = slope * line_x + intercept
    line_trace = go.Scatter(
        x=line_x,
        y=line_y,
        mode='lines',
        line=dict(dash='dot', color='orange'),
        name='Regression Line'
    )
    # Build the figure
    fig = go.Figure(data=[scatter_trace, line_trace])
    fig.update_layout(
        title='Portfolio Return wrt SP500',
        xaxis_title='SP500 log return',
        yaxis_title='Portfolio log return'
    )

    # 6) Add the regression equation and R^2 on the chart
    eq_text = f'y = {slope:.4f}x + {intercept:.4f}<br>R² = {r2:.4f}'
    fig.add_annotation(
        x=0.05, y=0.95,
        xref='paper', yref='paper',
        text=eq_text,
        showarrow=False,
        font=dict(size=14, color='black')
    )

    return fig

# Below is the method I follow the class exercise, which uses cumulative return as the calculation method to get the regrssion line. 
# If you want to run the following code, just remove the comments.

# def update_regression_scatter(n_clicks, risk_tolerance, stock_ticker):
#     """
#     Calculate the linear regression between the overall portfolio and SP500 daily cummulative returns,
#     and plot the scatter with the regression line.
#     """
#     # 1) Get the portfolio value series
#     Allocated, InvestmentReturn = get_asset_allocation(risk_tolerance, stock_ticker)
#     portfolio_val = InvestmentReturn.iloc[:, 0].dropna()
    
#     # 2) Read the SP500 benchmark data, get the value series
#     try:
#         benchmark_df = pd.read_csv('SP500Index.csv', index_col=0)
#         benchmark_df.index = pd.to_datetime(benchmark_df.index)
#         benchmark_df = benchmark_df.sort_index()
#         sp500_price = benchmark_df.iloc[:, 0].dropna()
#     except FileNotFoundError:
#         start_date = portfolio_val.index.min().strftime('%Y-%m-%d')
#         end_date = portfolio_val.index.max().strftime('%Y-%m-%d')
#         benchmark_df = yf.download('^GSPC', start=start_date, end=end_date)
#         if 'Close' in benchmark_df.columns:
#             sp500_price = benchmark_df['Close'].dropna()
#         else:
#             raise KeyError("'Close' not found in downloaded benchmark data")
    
#     # 3) Align the data
#     combined_val = pd.concat([portfolio_val, sp500_price], axis=1, join='inner')
#     combined_val.columns = ['portfolio', 'benchmark']
    
#     # 4) Using the first day after the merger as the baseline
#     # Here, cumulative yield = (day value - baseline value)/baseline value
#     combined_cum = (combined_val - combined_val.iloc[0]) / combined_val.iloc[0]
    
#     # Check if the data is sufficient for regression
#     if len(combined_cum) < 2:
#         return go.Figure()
    
#     # 5) Regression analysis：portfolio = intercept + slope * benchmark
#     slope, intercept = np.polyfit(combined_cum['benchmark'], combined_cum['portfolio'], 1)
#     predicted = slope * combined_cum['benchmark'] + intercept
#     r2 = r2_score(combined_cum['portfolio'], predicted)
    
#     # 6) Scatter and regression line traces
#     scatter_trace = go.Scatter(
#         x=combined_cum['benchmark'],
#         y=combined_cum['portfolio'],
#         mode='markers',
#         name='Cumulative Returns'
#     )

#     x_min, x_max = combined_cum['benchmark'].min(), combined_cum['benchmark'].max()
#     line_x = np.linspace(x_min, x_max, 50)
#     line_y = slope * line_x + intercept
#     line_trace = go.Scatter(
#         x=line_x,
#         y=line_y,
#         mode='lines',
#         line=dict(dash='dot', color='orange'),
#         name='Regression Line'
#     )
    
#     fig = go.Figure(data=[scatter_trace, line_trace])
#     fig.update_layout(
#         title='Portfolio Cumulative Return vs SP500 Cumulative Return',
#         xaxis_title='SP500 Cumulative Return',
#         yaxis_title='Portfolio Cumulative Return'
#     )
    
#     eq_text = f'y = {slope:.4f}x + {intercept:.4f}<br>R² = {r2:.4f}'
#     fig.add_annotation(
#         x=0.05, y=0.95,
#         xref='paper', yref='paper',
#         text=eq_text,
#         showarrow=False,
#         font=dict(size=14, color='black')
#     )
    
#     return fig


if __name__ == '__main__':
    app.run_server()
