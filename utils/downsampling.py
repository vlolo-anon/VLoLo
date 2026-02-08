import torch
import torch.nn.functional as F
import numpy as np
from typing import Union


class GPUDownsampler:
    """GPU-accelerated downsampler supporting multiple methods."""

    def __init__(self, device='cuda:0'):
        """
        Args:
            device: Device to use (default: 'cuda:0')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'

    def downsample(self, data: torch.Tensor, target_length: int,
                   method: str = 'linear_interpolation') -> torch.Tensor:

        data = data.to(self.device)

        squeeze_last = False
        if data.dim() == 2:
            data = data.unsqueeze(-1)
            squeeze_last = True

        B, L_long, N = data.shape
        L_short = target_length

        if L_long == L_short:
            if squeeze_last:
                return data.squeeze(-1)
            return data

        if method == 'linear_interpolation':
            downsampled = self._downsample_linear(data, L_short)
        elif method == 'spline_interpolation':
            downsampled = self._downsample_linear(data, L_short)
        elif method == 'mean':
            downsampled = self._downsample_mean(data, L_short)
        elif method == 'max':
            downsampled = self._downsample_max(data, L_short)
        elif method == 'min':
            downsampled = self._downsample_min(data, L_short)
        elif method == 'median':
            downsampled = self._downsample_median(data, L_short)
        elif method == 'first':
            downsampled = self._downsample_first(data, L_short)
        elif method == 'last':
            downsampled = self._downsample_last(data, L_short)
        else:
            print(f"Warning: Unknown downsample method '{method}', using linear_interpolation")
            downsampled = self._downsample_linear(data, L_short)

        if squeeze_last:
            downsampled = downsampled.squeeze(-1)

        return downsampled

    def _downsample_linear(self, data: torch.Tensor, target_length: int) -> torch.Tensor:
        """Linear interpolation downsampling (fastest, recommended)."""
        data_reshaped = data.permute(0, 2, 1)  # (B, N, L_long)
        downsampled_reshaped = F.interpolate(
            data_reshaped,
            size=target_length,
            mode='linear',
            align_corners=True
        )
        downsampled = downsampled_reshaped.permute(0, 2, 1)  # (B, L_short, N)
        return downsampled

    def _downsample_mean(self, data: torch.Tensor, target_length: int) -> torch.Tensor:
        """Average pooling downsampling."""
        data_reshaped = data.permute(0, 2, 1)
        downsampled_reshaped = F.adaptive_avg_pool1d(data_reshaped, target_length)
        downsampled = downsampled_reshaped.permute(0, 2, 1)
        return downsampled

    def _downsample_max(self, data: torch.Tensor, target_length: int) -> torch.Tensor:
        """Max pooling downsampling."""
        data_reshaped = data.permute(0, 2, 1)
        downsampled_reshaped = F.adaptive_max_pool1d(data_reshaped, target_length)
        downsampled = downsampled_reshaped.permute(0, 2, 1)
        return downsampled

    def _downsample_min(self, data: torch.Tensor, target_length: int) -> torch.Tensor:
        """Min pooling downsampling (via negated max pooling)."""
        data_reshaped = data.permute(0, 2, 1)
        downsampled_reshaped = -F.adaptive_max_pool1d(-data_reshaped, target_length)
        downsampled = downsampled_reshaped.permute(0, 2, 1)
        return downsampled

    def _downsample_median(self, data: torch.Tensor, target_length: int) -> torch.Tensor:
        """Median downsampling (slower due to CPU processing)."""
        B, L_long, N = data.shape
        L_short = target_length
        segment_size = L_long // L_short

        downsampled = torch.zeros(B, L_short, N, device=self.device)

        for i in range(L_short):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size if i < L_short - 1 else L_long
            segment = data[:, start_idx:end_idx, :]
            downsampled[:, i, :] = torch.median(segment, dim=1).values

        return downsampled

    def _downsample_first(self, data: torch.Tensor, target_length: int) -> torch.Tensor:
        """Take first value of each segment (uniform sampling)."""
        B, L_long, N = data.shape
        L_short = target_length
        indices = torch.linspace(0, L_long - 1, L_short, device=self.device).long()
        downsampled = data[:, indices, :]
        return downsampled

    def _downsample_last(self, data: torch.Tensor, target_length: int) -> torch.Tensor:
        """Take last value of each segment."""
        B, L_long, N = data.shape
        L_short = target_length
        segment_size = L_long // L_short

        indices = torch.arange(L_short, device=self.device) * segment_size + (segment_size - 1)
        indices = torch.clamp(indices, max=L_long - 1)
        downsampled = data[:, indices, :]

        return downsampled


def downsample_time_series_gpu(data: Union[torch.Tensor, np.ndarray],
                               target_length: int,
                               method: str = 'linear_interpolation',
                               device: str = None) -> torch.Tensor:

    if device is None:
        device = data.device if isinstance(data, torch.Tensor) else 'cuda:0'

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    elif not isinstance(data, torch.Tensor):
        data = torch.tensor(data).float()

    downsampler = GPUDownsampler(device=device)
    downsampled = downsampler.downsample(data, target_length, method)

    return downsampled