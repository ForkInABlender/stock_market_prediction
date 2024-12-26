# Dylan Kenneth Eliot

"""
This is the pytorch example, which the pybrain3 on is templated from.

Both have been tested manually. This one, because torch is bloatware, will not be used in production code, unlike pybrain3.

"""


import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# 3. Dataset and DataLoader
class StockDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# 4. PyTorch Model
class StockPredictor(nn.Module):
    def __init__(self, input_size):
        super(StockPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# 5. Main Script
if __name__ == "__main__":
    # Download and preprocess data
    ticker = "AAPL"
    data = download_data(ticker, "2020-01-01", "2023-01-01")
    data = generate_features(data)

    # Prepare inputs and targets
    features = data[['SMA_10', 'SMA_50', 'RSI']].values
    targets = data['Target'].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    train_dataset = StockDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = StockDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model, Loss, Optimizer
    model = StockPredictor(input_size=3)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    for epoch in range(10):  # Increase epochs for better results
        model.train()
        epoch_loss = 0
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features).squeeze()
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            outputs = model(batch_features).squeeze()
            predictions = (outputs > 0.5).float()
            correct += (predictions == batch_targets).sum().item()
            total += batch_targets.size(0)
    print(f"Accuracy: {correct / total:.2%}")
