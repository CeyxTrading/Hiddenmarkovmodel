import os
import numpy as np
import pandas as pd
from pyhhmm.gaussian import GaussianHMM
from pyhhmm.utils import save_model, load_model
import pyhhmm.utils as hu
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests


#  TIINGO API Key - todo
TIINGO_API_KEY = "<Your Tiingo API Key>"

#  Set up directories
results_path = "C:\\dev\\trading\\tradesystem1\\results\\market_trends_hidden_markov"


def create_output_dirs():
    os.makedirs(results_path, exist_ok=True)


def fetch_crypto(symbol, start_date, end_date, interval):
    try:
        fetch_url = f"https://api.tiingo.com/tiingo/crypto/prices?tickers={symbol}&startDate={start_date}&endDate={end_date}&resampleFreq={interval}&columns=date,open,high,low,close,volume&token={TIINGO_API_KEY}"
        headers = {
            'Accept': 'application/json'
        }

        #  Send GET request
        response = requests.get(fetch_url, headers=headers)
        data = response.json()

        data_df = pd.DataFrame()
        list_item = data[0]
        price_data = list_item['priceData']
        for row in price_data:
            row_df = pd.DataFrame({'date': [row['date']], 'open': [row['open']], 'high': [row['high']], 'low': [row['low']], 'close': [row['close']], 'volume': [row['volume']]})
            data_df = pd.concat([data_df, row_df], axis=0, ignore_index=True)

        return data_df
    except Exception as e:
        print(f"Failed to fetch stock data for {symbol}, error: {str(e)}")
        return None


def add_metrics(price_df):
    #  Add percentage price change since yesterday
    price_df['Change'] = price_df['close'].pct_change(1)

    #  Add volatility
    price_df['Volatility'] = (price_df['high'] / price_df['low']) - 1

    price_df.dropna(inplace=True)

    return price_df


def create_market_states(states, price_list):
    state_1 = []
    state_2 = []
    state_3 = []

    i = 0
    for state in states:
        if state == 0:
            state_1.append(price_list[i])
            state_2.append(np.nan)
            state_3.append(np.nan)
        elif state == 1:
            state_1.append(np.nan)
            state_2.append(price_list[i])
            state_3.append(np.nan)
        elif state == 2:
            state_1.append(np.nan)
            state_2.append(np.nan)
            state_3.append(price_list[i])
        i += 1

    return state_1, state_2, state_3


def plot_market_states(state_1, state_2, state_3):
    light_palette = {}
    light_palette["bg_color"] = "#ffffff"
    light_palette["plot_bg_color"] = "#ffffff"
    light_palette["grid_color"] = "#e6e6e6"
    light_palette["text_color"] = "#2e2e2e"
    light_palette["border_color"] = "#2e2e2e"

    palette = light_palette

    #  Create sub plots
    fig = make_subplots(rows=1, cols=1, subplot_titles=[f"Market States for {symbol}"])

    #  Plot states
    index = np.arange(0, len(state_1))
    fig.add_trace(go.Scatter(
        name='State 1',
        x=index, y=state_1, marker_color="blue"), row=1, col=1)

    fig.add_trace(go.Scatter(
        name='State 2',
        x=index, y=state_2, marker_color="green"), row=1, col=1)

    fig.add_trace(go.Scatter(
        name='State 3',
        x=index, y=state_3, marker_color="red"), row=1, col=1)


    fig.update_layout(
        title={'text': '', 'x': 0.5},
        font=dict(family="Verdana", size=12, color=palette["text_color"]),
        autosize=True,
        width=1280, height=720,
        xaxis={"rangeslider": {"visible": False}},
        plot_bgcolor=palette["plot_bg_color"],
        paper_bgcolor=palette["bg_color"])
    fig.update_yaxes(visible=False, secondary_y=True)

    #  Change grid color
    fig.update_xaxes(showline=True, linewidth=1, linecolor=palette["grid_color"], gridcolor=palette["grid_color"])
    fig.update_yaxes(showline=True, linewidth=1, linecolor=palette["grid_color"], gridcolor=palette["grid_color"])

    #  Create output file
    file_name = f"{symbol}_market_states.png"
    path = os.path.join(results_path, file_name)
    fig.write_image(path, format="png")


def load_trained_model():
    file_name = "market_trends_hidden_markov1.pkl"
    if os.path.exists(file_name):
        model = load_model(file_name)
        return model


def save_trained_model(model):
    file_name = "market_trends_hidden_markov1.pkl"
    save_model(model, file_name)


if __name__ == '__main__':
    create_output_dirs()

    symbol = 'btcusd'
    interval = '15min'
    start_date = '2022-09-01'
    end_date = '2022-10-01'
    price_df = fetch_crypto(symbol, start_date, end_date, interval)
    if price_df is None or price_df.empty:
        print(f"No data available for {symbol}")
        exit(0)

    #  Add change and volatility
    price_df = add_metrics(price_df)

    #  Create train/test data
    x_train = pd.DataFrame(price_df[['Change', 'Volatility']])

    #  Try to load model
    model = load_trained_model()
    if model is None:
        #  Create model
        model = GaussianHMM(n_states=3,
                            n_emissions=2,
                            covariance_type='full',
                            verbose=True)

        #  Train model
        model, log_likelihoods = model.train([np.array(x_train.values)])

        #  Display trained model results
        hu.pretty_print_hmm(model, hmm_type='Gaussian')

    #  Make predictions based on training data
    states = model.predict([x_train.values])[0]

    #  Print hidden states
    for i in range(3):
        print("{0}th hidden state".format(i))
        print("mean = ", model.means[i])
        print("var = ", np.diag(model.covars[i]))
        print()

    #  Assign prices into the different market states
    price_list = price_df['close'].tolist()
    state_1, state_2, state_3 = create_market_states(states, price_list)

    #  Plot market states
    plot_market_states(state_1, state_2, state_3)

    #  Save model
    save_trained_model(model)