# Dylan Kenneth Eliot

"""
This allows for a higher degree of accuracy via pybrain3 on what you're going to get, win or lose, on stock market value.

"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pybrain3.datasets import SupervisedDataSet
from pybrain3.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from pybrain3.supervised.trainers import BackpropTrainer

# 1. Data Collection
def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# 2. Feature Engineering
def generate_features(data):
    data['SMA_10'] = data['Close'].rolling(window=10).mean()  # 10-day SMA
    data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day SMA
    data['RSI'] = compute_rsi(data['Close'])  # Relative Strength Index
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)  # 1 for buy, 0 for sell
    data = data.dropna()
    return data

def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 3. Building the PyBrain Neural Network
def build_network(input_size, hidden_size, output_size):
    net = FeedForwardNetwork()
    in_layer = LinearLayer(input_size)
    hidden_layer = SigmoidLayer(hidden_size)
    out_layer = SigmoidLayer(output_size)

    # Connect layers
    net.addInputModule(in_layer)
    net.addModule(hidden_layer)
    net.addOutputModule(out_layer)

    in_to_hidden = FullConnection(in_layer, hidden_layer)
    hidden_to_out = FullConnection(hidden_layer, out_layer)

    net.addConnection(in_to_hidden)
    net.addConnection(hidden_to_out)

    net.sortModules()
    return net

# 4. Training the Network
def train_network(net, dataset, epochs=10, learning_rate=0.01):
    trainer = BackpropTrainer(net, dataset, learningrate=learning_rate, verbose=True)
    for epoch in range(epochs):
        trainer.train()

# Main Script
if __name__ == "__main__":
    # Download and preprocess data
    ticker = "BTC-USD"
    data = download_data(ticker, "2020-01-01", "2023-01-01")
    data = generate_features(data)

    # Prepare inputs and targets
    features = data[['SMA_10', 'SMA_50', 'RSI']].values
    targets = data['Target'].values.reshape(-1, 1)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

    # Convert to PyBrain dataset
    train_dataset = SupervisedDataSet(X_train.shape[1], 1)
    for x, y in zip(X_train, y_train):
        train_dataset.addSample(x, y)

    test_dataset = SupervisedDataSet(X_test.shape[1], 1)
    for x, y in zip(X_test, y_test):
        test_dataset.addSample(x, y)

    # Build and train the network
    input_size = X_train.shape[1]
    hidden_size = 8  # Arbitrary, can be tuned
    output_size = 1
    net = build_network(input_size, hidden_size, output_size)
    train_network(net, train_dataset, epochs=20, learning_rate=0.01)

    # Evaluate the network
    correct = 0
    total = 0
    for x, y in zip(X_test, y_test):
        predicted = net.activate(x)
        prediction = 1 if predicted > 0.5 else 0
        correct += (prediction == y[0])
        total += 1
    print(f"Accuracy: {correct / total:.2%}")
