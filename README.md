We've added these models to all our bots to help determine hidden states of the market. Also lowers volume of trades, increasing profit rate. 

If you are unfamiliar with Hidden Markov Models and/or are unaware of how they can be used as a risk management tool read here https://hmmlearn.readthedocs.io/en/0.2.0/auto_examples/plot_hmm_stock_analysis.html

Helped me this year determine:

Upward trending state

Upward breakout state

Downward trending state

Downward plunge state

Ranging state

Requirements
numpy
pandas
pyhhmm
requests
plotly==5.10.0
kaleido==0.1.0.post1

How to do it:
Obtain OHLC (open, high, low, close) price data for a specific time period using the Tiingo API or another data source. 

Calculate the one-day price change and high/low range for the data. 

Create a Gaussian Hidden Markov Model (HMM) and train it using the price change and high/low range data. 

Use the trained HMM to make predictions for hidden states. Iterate through the predicted states and assign prices for each time period to each state. 

Create a price plot by plotting each state in a different color.
