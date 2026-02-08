import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple


class GPUTimeSeriesFeatureExtractor:

    def __init__(self, device='cuda:0'):

        self.device = device if torch.cuda.is_available() else 'cpu'

    def extract_trend_features_gpu(self, data: torch.Tensor, window_ratio: float = 0.1) -> torch.Tensor:

        *batch_dims, time_len = data.shape
        window_size = max(3, int(time_len * window_ratio))
        if window_size % 2 == 0:
            window_size += 1

        data_2d = data.reshape(-1, time_len).unsqueeze(1)
        kernel = torch.ones(1, 1, window_size, device=self.device) / window_size
        padding = window_size // 2

        trend_2d = F.conv1d(data_2d, kernel, padding=padding)
        trend = trend_2d.squeeze(1).reshape(*batch_dims, time_len)

        return trend

    def extract_derivative_features_gpu(self, data: torch.Tensor, order: int = 1) -> torch.Tensor:

        if order == 1:
            derivative = torch.diff(data, dim=-1, prepend=data[..., :1])
        elif order == 2:
            first_diff = torch.diff(data, dim=-1, prepend=data[..., :1])
            derivative = torch.diff(first_diff, dim=-1, prepend=first_diff[..., :1])
        else:
            raise ValueError("order must be 1 or 2")

        return derivative

    def extract_amplitude_features_gpu(self, data: torch.Tensor) -> torch.Tensor:

        *batch_dims, time_len = data.shape
        window_size = max(5, time_len // 20)
        if window_size % 2 == 0:
            window_size += 1

        abs_data = torch.abs(data)
        abs_data_2d = abs_data.reshape(-1, time_len).unsqueeze(1)

        padding = window_size // 2
        amplitude_2d = F.max_pool1d(
            F.pad(abs_data_2d, (padding, padding), mode='replicate'),
            kernel_size=window_size,
            stride=1
        )

        amplitude = amplitude_2d.squeeze(1).reshape(*batch_dims, time_len)

        return amplitude

    def extract_frequency_features_gpu(self, data: torch.Tensor, split_ratio: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:

        *batch_dims, time_len = data.shape

        fft_data = torch.fft.fft(data, dim=-1)
        freqs = torch.fft.fftfreq(time_len, device=self.device)

        nyquist = 0.5
        split_freq = nyquist * split_ratio

        freq_mask_low = torch.abs(freqs) <= split_freq
        freq_mask_high = ~freq_mask_low

        fft_low = fft_data * freq_mask_low
        fft_high = fft_data * freq_mask_high

        low_freq = torch.fft.ifft(fft_low, dim=-1).real
        high_freq = torch.fft.ifft(fft_high, dim=-1).real

        return low_freq, high_freq

    def extract_features_gpu(self, data: torch.Tensor, enable_fft: bool = True,
                             feature_options: list = None) -> torch.Tensor:

        data = data.to(self.device)

        if data.dim() == 2:
            data = data.unsqueeze(-1)

        batch_size, time_len, n_variables = data.shape

        if feature_options is None:
            feature_options = [0, 1, 2, 3, 4, 5, 6]

        n_features = len(feature_options)

        # Flatten all variables for batch processing: (B, T, N) -> (B*N, T)
        data_flat = data.permute(0, 2, 1).reshape(batch_size * n_variables, time_len)

        features_flat = torch.zeros(batch_size * n_variables, n_features, time_len, device=self.device)

        # Pre-compute shared features
        need_trend = 0 in feature_options or 1 in feature_options
        need_fft = enable_fft and (5 in feature_options or 6 in feature_options)

        if need_trend:
            trend_flat = self.extract_trend_features_gpu(data_flat)

        if need_fft:
            low_freq_flat, high_freq_flat = self.extract_frequency_features_gpu(data_flat)

        # Extract selected features
        for f_idx, feature_id in enumerate(feature_options):
            if feature_id == 0:
                features_flat[:, f_idx, :] = trend_flat
            elif feature_id == 1:
                features_flat[:, f_idx, :] = data_flat - trend_flat
            elif feature_id == 2:
                features_flat[:, f_idx, :] = self.extract_derivative_features_gpu(data_flat, order=1)
            elif feature_id == 3:
                features_flat[:, f_idx, :] = self.extract_derivative_features_gpu(data_flat, order=2)
            elif feature_id == 4:
                features_flat[:, f_idx, :] = self.extract_amplitude_features_gpu(data_flat)
            elif feature_id == 5 and enable_fft:
                features_flat[:, f_idx, :] = low_freq_flat
            elif feature_id == 6 and enable_fft:
                features_flat[:, f_idx, :] = high_freq_flat
            else:
                features_flat[:, f_idx, :] = 0.0

        # Reshape back: (B*N, F, T) -> (B, N, F, T)
        features = features_flat.reshape(batch_size, n_variables, n_features, time_len)

        return features


def extract_time_series_features_gpu(data: Union[torch.Tensor, np.ndarray],
                                     device: str = None,
                                     enable_fft: bool = True,
                                     feature_options: list = None) -> torch.Tensor:

    if device is None:
        device = data.device if isinstance(data, torch.Tensor) else 'cuda:0'

    if feature_options is None:
        feature_options = [0, 1, 2, 3, 4, 5, 6]

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    elif not isinstance(data, torch.Tensor):
        data = torch.tensor(data).float()

    extractor = GPUTimeSeriesFeatureExtractor(device=device)
    features = extractor.extract_features_gpu(data, enable_fft=enable_fft, feature_options=feature_options)

    return features