from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
from torch.nn import functional as F


class ContrastiveLoss(nn.Module):
    """
    Triplet margin loss for memory separation.
    Encourages queries to be closer to nearest memory item than second nearest.
    """

    def __init__(self, temp_param, eps=1e-12, reduce=True):
        super(ContrastiveLoss, self).__init__()
        self.temp_param = temp_param
        self.eps = eps
        self.reduce = reduce

    def get_score(self, query, key):
        """
        Compute attention scores between queries and memory items.

        Args:
            query: (T, C) initial latent features
            key: (M, C) memory items

        Returns:
            score: (T, M) attention scores
        """
        score = torch.matmul(query, torch.t(key))
        score = F.softmax(score, dim=1)
        return score

    def forward(self, queries, items):
        """
        Compute triplet margin loss.

        Args:
            queries: (N, L, C) query features
            items: (M, C) memory items

        Returns:
            loss: Triplet margin loss
        """
        batch_size = queries.size(0)
        d_model = queries.size(-1)

        loss = torch.nn.TripletMarginLoss(margin=1.0, reduce=self.reduce)

        queries = queries.contiguous().view(-1, d_model)
        score = self.get_score(queries, items)

        # Get indices of nearest and second nearest items
        _, indices = torch.topk(score, 2, dim=1)

        # 1st and 2nd nearest items (anchor, positive, negative)
        pos = items[indices[:, 0]]
        neg = items[indices[:, 1]]
        anc = queries

        spread_loss = loss(anc, pos, neg)

        if self.reduce:
            return spread_loss

        spread_loss = spread_loss.contiguous().view(batch_size, -1)
        return spread_loss


class GatheringLoss(nn.Module):
    """
    Gathering loss for memory compactness.
    Encourages queries to be close to their nearest memory item.
    Supports variable-wise memory banks.
    """

    def __init__(self, reduce=True):
        super(GatheringLoss, self).__init__()
        self.reduce = reduce

    def get_score(self, query, key):
        """
        Compute attention scores.

        Args:
            query: (T, C) latent features
            key: (M, C) memory items for specific variable

        Returns:
            score: (T, M) attention scores
        """
        score = torch.matmul(query, torch.t(key))
        score = F.softmax(score, dim=1)
        return score

    def forward(self, queries, items):
        """
        Compute gathering loss for each variable independently.

        Args:
            queries: (B, N, C) batch size, num variables, feature dim
            items: (N, M, C) num variables, num memory slots, feature dim

        Returns:
            loss: Gathering loss (scalar if reduce=True, else per-sample loss)
        """
        batch_size, n_vars, d_model = queries.shape
        _, n_memory, _ = items.shape

        if self.reduce:
            loss_mse = torch.nn.MSELoss(reduction='mean')
        else:
            loss_mse = torch.nn.MSELoss(reduction='none')

        all_losses = []

        for var_idx in range(n_vars):
            # Get queries and memory items for this variable
            queries_var = queries[:, var_idx, :]  # (B, C)
            items_var = items[var_idx, :, :]  # (M, C)

            # Compute attention scores
            score = self.get_score(queries_var, items_var)  # (B, M)

            # Select nearest memory item
            _, indices = torch.topk(score, 1, dim=1)  # (B, 1)

            # Compute MSE loss with selected memory item
            selected_items = items_var[indices.squeeze(1)]  # (B, C)
            gathering_loss_var = loss_mse(queries_var, selected_items)

            if not self.reduce:
                if gathering_loss_var.dim() > 1:
                    gathering_loss_var = torch.sum(gathering_loss_var, dim=-1)

            all_losses.append(gathering_loss_var)

        if self.reduce:
            return torch.mean(torch.stack(all_losses))
        else:
            if n_vars == 1:
                return all_losses[0]
            else:
                return torch.stack(all_losses, dim=1)


class EntropyLoss(nn.Module):
    """
    Entropy loss for attention sparsity.
    Encourages sparse attention weights over memory items.
    """

    def __init__(self, eps=1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, x):
        """
        Compute entropy loss.

        Args:
            x: (T, M) attention weights

        Returns:
            loss: Mean entropy loss
        """
        loss = -1 * x * torch.log(x + self.eps)
        loss = torch.sum(loss, dim=-1)
        loss = torch.mean(loss)
        return loss