A hidden Markov model (HMM) is a statistical model that can be used to analyze and predict the likelihood of certain sequences of events. In the context of crypto trading, an HMM could potentially be used to predict price movements or other market trends based on past data.

One potential application of an HMM in crypto trading is to analyze historical data on the prices of different cryptocurrencies and use this information to predict future price movements. This could involve training an HMM on a large dataset of historical price data, and then using the trained model to make predictions about the future price of a particular cryptocurrency.

There are several potential benefits to using an HMM for crypto trading. For example, an HMM can take into account the dependencies between different events, such as the relationship between the prices of different cryptocurrencies or the impact of market news on price movements. Additionally, an HMM can be used to analyze data from multiple sources, such as social media posts or news articles, to inform its predictions

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
