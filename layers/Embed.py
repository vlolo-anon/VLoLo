import torch
import torch.nn as nn
import math


class DataEmbedding_inverted(nn.Module):
    """Inverted data embedding for time series."""

    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Args:
            x: (B, L, N) - Batch, Time, Variate

        Returns:
            (B, N, E) - Batch, Variate, Embedding
        """
        x = x.permute(0, 2, 1)  # (B, N, L)
        x = self.value_embedding(x)
        return self.dropout(x)


class FeatureEmbedding_inverted(nn.Module):
    """Inverted feature embedding for time series features."""

    def __init__(self, seq_len, n_features, d_model, dropout=0.1):
        super(FeatureEmbedding_inverted, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model

        # Batched weights: (n_features, seq_len, d_model)
        self.weight = nn.Parameter(torch.empty(n_features, seq_len, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))

        self._init_weights()
        self.dropout = nn.Dropout(p=dropout)

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for f in range(self.n_features):
            nn.init.kaiming_uniform_(self.weight[f], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[f])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias[f], -bound, bound)

    def forward(self, x):
        """
        Args:
            x: (B, N, F, L) - Batch, Variate, Feature, Time

        Returns:
            (B, N, F, E) - Batch, Variate, Feature, Embedding
        """
        # Batched einsum: (B, N, F, L) x (F, L, E) -> (B, N, F, E)
        embedded = torch.einsum('bvft,ftd->bvfd', x, self.weight)
        embedded = embedded + self.bias  # Broadcasting: (F, E) -> (B, N, F, E)

        return self.dropout(embedded)