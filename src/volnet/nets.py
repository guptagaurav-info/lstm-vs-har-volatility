"""PyTorch sequence models: a compact LSTM and a temporal convolutional network (TCN), trained on CPU."""
import random

import numpy as np
import torch
from torch import nn

torch.set_num_threads(2)


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.use_deterministic_algorithms(True)


class LSTMNet(nn.Module):
    def __init__(self, n_in, hidden=32, layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=layers, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                      # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)   # last hidden state -> one number


class Chomp(nn.Module):
    def __init__(self, k):
        super().__init__(); self.k = k

    def forward(self, x):
        return x[:, :, :-self.k] if self.k else x   # trim the right padding so the convolution stays causal


class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation), Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation), Chomp(pad), nn.ReLU(), nn.Dropout(dropout))
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return torch.relu(self.net(x) + self.skip(x))


class TCNNet(nn.Module):
    def __init__(self, n_in, channels=32, levels=4, kernel=3, dropout=0.1):
        super().__init__()
        layers, c = [], n_in
        for i in range(levels):                # dilations 1, 2, 4, 8: receptive field covers ~60 days
            layers.append(TCNBlock(c, channels, kernel, 2 ** i, dropout)); c = channels
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(channels, 1)

    def forward(self, x):                      # x: (batch, seq_len, features)
        h = self.tcn(x.transpose(1, 2))        # Conv1d wants (batch, channels, time)
        return self.head(h[:, :, -1]).squeeze(-1)


def train(model, X, y, epochs, lr, batch, weight_decay, seed):
    """Plain mini-batch training with MSE on the log target. Returns the fitted model."""
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xt[idx]), yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


@torch.no_grad()
def predict(model, X):
    model.eval()
    return model(torch.tensor(X, dtype=torch.float32)).numpy()
