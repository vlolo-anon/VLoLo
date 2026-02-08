import torch
import os
import random
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import signal, stats
from statsmodels.tsa.stattools import acf
import warnings

from utils.time_series_features_extractor import extract_time_series_features_gpu
from utils.downsampling import downsample_time_series_gpu


warnings.filterwarnings('ignore')


def _compute_window_count(data_len, win_size, step):

    if data_len < win_size:
        return 0
    
    # Normal windows count
    normal_windows = (data_len - win_size) // step + 1
    
    # Check if we need an extra window for the tail
    last_normal_end = (normal_windows - 1) * step + win_size
    if last_normal_end < data_len:
        return normal_windows + 1
    else:
        return normal_windows


def _compute_window_start(index, data_len, win_size, step):

    # Normal windows count
    normal_windows = (data_len - win_size) // step + 1
    last_normal_end = (normal_windows - 1) * step + win_size
    
    # Check if this is the extra tail window
    if index == normal_windows and last_normal_end < data_len:
        # Tail window: align to the end
        return data_len - win_size
    else:
        return index * step


def create_batch_feature_collate_fn(feature_indices=None, enable_fft=True, device='cuda:0',
                                     seq_len=None, long_term_multiplier=None,
                                     downsample_method='linear_interpolation'):

    if feature_indices is None:
        feature_indices = [0, 1, 2, 3, 4, 5, 6]

    enable_downsampling = (seq_len is not None and long_term_multiplier is not None
                           and long_term_multiplier > 1)

    def batch_feature_collate_fn(batch):

        try:
            data_list = []
            labels_list = []
            start_indices_list = []
            has_start_indices = len(batch[0]) == 3

            for item in batch:
                if len(item) == 2:
                    data, labels = item
                else:
                    data, labels, start_idx = item
                    start_indices_list.append(start_idx)

                data_list.append(data)
                labels_list.append(labels)

            # Convert to tensors
            data_tensors = []
            labels_tensors = []

            for data, labels in zip(data_list, labels_list):
                if isinstance(data, np.ndarray):
                    data_tensor = torch.from_numpy(data).float()
                else:
                    data_tensor = data.float()

                if isinstance(labels, np.ndarray):
                    labels_tensor = torch.from_numpy(labels).float()
                else:
                    labels_tensor = labels.float()

                if data_tensor.dim() == 1:
                    data_tensor = data_tensor.unsqueeze(-1)

                data_tensors.append(data_tensor)
                labels_tensors.append(labels_tensor)

            data_batch = torch.stack(data_tensors)
            labels_batch = torch.stack(labels_tensors)

            # start_indices_batch: only for test mode
            if has_start_indices:
                start_indices_batch = torch.tensor(start_indices_list, dtype=torch.long)
            else:
                start_indices_batch = None

            # Batch-wise feature extraction
            if len(feature_indices) > 0:
                time_features_batch = extract_time_series_features_gpu(
                    data_batch,
                    device=device,
                    enable_fft=enable_fft,
                    feature_options=feature_indices
                )
            else:
                B, L, C = data_batch.shape
                F = len(feature_indices)
                time_features_batch = torch.zeros(B, C, F, L)

            # Downsampling
            if enable_downsampling:
                downsampled_batch = downsample_time_series_gpu(
                    data_batch,
                    target_length=seq_len,
                    method=downsample_method,
                    device=device
                )
            else:
                downsampled_batch = None

            return data_batch, labels_batch, time_features_batch, downsampled_batch, start_indices_batch

        except Exception as e:
            print(f"Batch collate error: {e}")
            import traceback
            traceback.print_exc()

            # Fallback
            data_batch = torch.stack([torch.from_numpy(item[0]).float() for item in batch])
            labels_batch = torch.stack([torch.from_numpy(item[1]).float() for item in batch])

            has_start_indices = len(batch[0]) == 3
            if has_start_indices:
                start_indices_batch = torch.tensor(
                    [item[2] for item in batch], dtype=torch.long
                )
            else:
                start_indices_batch = None

            if data_batch.dim() == 2:
                data_batch = data_batch.unsqueeze(-1)

            B, L, C = data_batch.shape
            F = len(feature_indices) if feature_indices else 7
            time_features_batch = torch.zeros(B, C, F, L)

            if enable_downsampling:
                downsampled_batch = downsample_time_series_gpu(
                    data_batch, seq_len, downsample_method, device
                )
            else:
                downsampled_batch = None

            return data_batch, labels_batch, time_features_batch, downsampled_batch, start_indices_batch

    return batch_feature_collate_fn


class PeriodEstimator:
    """Period estimation using ACF (for UCR_Anomaly dataset only)."""

    def __init__(self, max_lag_ratio=0.25, min_period=3, max_period=None, alpha=0.05):
        self.max_lag_ratio = max_lag_ratio
        self.min_period = min_period
        self.max_period = max_period
        self.alpha = alpha

    def compute_autocorrelation(self, data, max_lags=None, method='statsmodels'):
        n = len(data)

        if max_lags is None:
            max_lags = min(n // 4, int(n * self.max_lag_ratio))
        else:
            max_lags = min(max_lags, n - 1)

        lags = np.arange(0, max_lags + 1)

        if method == 'statsmodels':
            try:
                autocorr, confint = acf(data, nlags=max_lags, alpha=0.05, fft=True)
                confidence_intervals = confint
                return lags, autocorr, confidence_intervals
            except Exception as e:
                print(f"Warning: statsmodels method failed ({e}), using numpy method")
                method = 'numpy'

        if method == 'numpy':
            data_centered = data - np.mean(data)
            autocorr = np.correlate(data_centered, data_centered, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
            autocorr = autocorr[:max_lags + 1]

        confidence_intervals = None
        if method == 'numpy':
            se = np.sqrt((1 + 2 * np.cumsum(autocorr[1:] ** 2)) / n)
            se = np.concatenate([np.array([1 / np.sqrt(n)]), se])[:len(autocorr)]
            confidence_intervals = np.column_stack([
                autocorr - 1.96 * se,
                autocorr + 1.96 * se
            ])

        return lags, autocorr, confidence_intervals

    def estimate_period_from_acf(self, data_length, lags, autocorr, min_lag=1):
        if len(autocorr) <= min_lag + 1:
            return None

        z = stats.norm.ppf(1 - self.alpha / 2.0)
        significance_level = z / np.sqrt(data_length)

        max_period = self.max_period if self.max_period else min(len(autocorr), data_length // 3)

        sub_acf = autocorr[min_lag:]
        peaks, properties = signal.find_peaks(sub_acf, height=significance_level)

        if len(peaks) == 0:
            return None

        peaks = peaks + min_lag
        valid_peaks = peaks[peaks <= max_period]

        if len(valid_peaks) == 0:
            return None

        best_peak = valid_peaks[np.argmax(autocorr[valid_peaks])]
        return int(best_peak)

    def estimate_periods_multivariate(self, data, max_lags=None, method='statsmodels'):
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_features = data.shape[1]
        print(f"Estimating single period for {n_features} features using ACF...")

        combined_data = np.mean(data, axis=1)

        if np.isnan(combined_data).any():
            combined_data = combined_data[~np.isnan(combined_data)]

        if len(combined_data) < self.min_period * 4:
            print("  Insufficient data for period estimation")
            return np.array([[0]])

        try:
            lags, autocorr, _ = self.compute_autocorrelation(combined_data, max_lags, method)
            period = self.estimate_period_from_acf(len(combined_data), lags, autocorr)

            if period is not None and period >= self.min_period:
                print(f"  Estimated period for combined data: {period}")
                return np.array([[period]])
            else:
                print("  No significant period detected in combined data")
                return np.array([[0]])

        except Exception as e:
            print(f"  Error in period estimation ({e}), period = 0")
            return np.array([[0]])


class SWaTSegLoader(Dataset):
    """SWaT dataset loader."""

    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data = pd.read_csv(data_path + 'swat_train2.csv', header=1)
        data = data.values[:, :-1]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        test_data = pd.read_csv(data_path + 'swat2.csv')
        y = test_data['Normal/Attack'].to_numpy()
        labels = []
        for i in y:
            if i == 1:
                labels.append(1)
            else:
                labels.append(0)
        labels = np.array(labels)
        print('anomaly num:', np.sum(labels))

        test_data = test_data.values[:, :-1]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)
        self.train = data
        self.test_labels = labels.reshape(-1, 1)

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return _compute_window_count(self.train.shape[0], self.win_size, self.step)
        elif self.mode == 'test':
            return _compute_window_count(self.test.shape[0], self.win_size, self.step)
        else:
            return _compute_window_count(self.train.shape[0], self.win_size, self.step)

    def __getitem__(self, index):
        if self.mode == "train":
            data_len = self.train.shape[0]
            start_idx = _compute_window_start(index, data_len, self.win_size, self.step)
            data = np.float32(self.train[start_idx:start_idx + self.win_size])
            labels = np.float32(self.test_labels[0:self.win_size])
            return (data, labels)
        elif self.mode == 'test':
            data_len = self.test.shape[0]
            start_idx = _compute_window_start(index, data_len, self.win_size, self.step)
            data = np.float32(self.test[start_idx:start_idx + self.win_size])
            labels = np.float32(self.test_labels[start_idx:start_idx + self.win_size])
            return (data, labels, start_idx)
        else:
            data_len = self.train.shape[0]
            start_idx = _compute_window_start(index, data_len, self.win_size, self.step)
            data = np.float32(self.train[start_idx:start_idx + self.win_size])
            labels = np.float32(self.test_labels[0:self.win_size])
            return (data, labels)


class PSMSegLoader(Dataset):
    """PSM dataset loader."""

    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        test_data = pd.read_csv(data_path + '/test.csv')
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)
        self.train = data
        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return _compute_window_count(self.train.shape[0], self.win_size, self.step)
        elif self.mode == 'test':
            return _compute_window_count(self.test.shape[0], self.win_size, self.step)
        else:
            return _compute_window_count(self.train.shape[0], self.win_size, self.step)

    def __getitem__(self, index):
        if self.mode == "train":
            data_len = self.train.shape[0]
            start_idx = _compute_window_start(index, data_len, self.win_size, self.step)
            data = np.float32(self.train[start_idx:start_idx + self.win_size])
            labels = np.float32(self.test_labels[0:self.win_size])
            return (data, labels)
        elif self.mode == 'test':
            data_len = self.test.shape[0]
            start_idx = _compute_window_start(index, data_len, self.win_size, self.step)
            data = np.float32(self.test[start_idx:start_idx + self.win_size])
            labels = np.float32(self.test_labels[start_idx:start_idx + self.win_size])
            return (data, labels, start_idx)
        else:
            data_len = self.train.shape[0]
            start_idx = _compute_window_start(index, data_len, self.win_size, self.step)
            data = np.float32(self.train[start_idx:start_idx + self.win_size])
            labels = np.float32(self.test_labels[0:self.win_size])
            return (data, labels)


class MSLSegLoader(Dataset):
    """MSL dataset loader."""

    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return _compute_window_count(self.train.shape[0], self.win_size, self.step)
        elif self.mode == 'test':
            return _compute_window_count(self.test.shape[0], self.win_size, self.step)
        else:
            return _compute_window_count(self.train.shape[0], self.win_size, self.step)

    def __getitem__(self, index):
        if self.mode == "train":
            data_len = self.train.shape[0]
            start_idx = _compute_window_start(index, data_len, self.win_size, self.step)
            data = np.float32(self.train[start_idx:start_idx + self.win_size])
            labels = np.float32(self.test_labels[0:self.win_size])
            return (data, labels)
        elif self.mode == 'test':
            data_len = self.test.shape[0]
            start_idx = _compute_window_start(index, data_len, self.win_size, self.step)
            data = np.float32(self.test[start_idx:start_idx + self.win_size])
            labels = np.float32(self.test_labels[start_idx:start_idx + self.win_size])
            return (data, labels, start_idx)
        else:
            data_len = self.train.shape[0]
            start_idx = _compute_window_start(index, data_len, self.win_size, self.step)
            data = np.float32(self.train[start_idx:start_idx + self.win_size])
            labels = np.float32(self.test_labels[0:self.win_size])
            return (data, labels)


class SMAPSegLoader(Dataset):
    """SMAP dataset loader."""

    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return _compute_window_count(self.train.shape[0], self.win_size, self.step)
        elif self.mode == 'test':
            return _compute_window_count(self.test.shape[0], self.win_size, self.step)
        else:
            return _compute_window_count(self.train.shape[0], self.win_size, self.step)

    def __getitem__(self, index):
        if self.mode == "train":
            data_len = self.train.shape[0]
            start_idx = _compute_window_start(index, data_len, self.win_size, self.step)
            data = np.float32(self.train[start_idx:start_idx + self.win_size])
            labels = np.float32(self.test_labels[0:self.win_size])
            return (data, labels)
        elif self.mode == 'test':
            data_len = self.test.shape[0]
            start_idx = _compute_window_start(index, data_len, self.win_size, self.step)
            data = np.float32(self.test[start_idx:start_idx + self.win_size])
            labels = np.float32(self.test_labels[start_idx:start_idx + self.win_size])
            return (data, labels, start_idx)
        else:
            data_len = self.train.shape[0]
            start_idx = _compute_window_start(index, data_len, self.win_size, self.step)
            data = np.float32(self.train[start_idx:start_idx + self.win_size])
            labels = np.float32(self.test_labels[0:self.win_size])
            return (data, labels)


class SMDSegLoader(Dataset):
    """SMD dataset loader."""

    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return _compute_window_count(self.train.shape[0], self.win_size, self.step)
        elif self.mode == 'test':
            return _compute_window_count(self.test.shape[0], self.win_size, self.step)
        else:
            return _compute_window_count(self.train.shape[0], self.win_size, self.step)

    def __getitem__(self, index):
        if self.mode == "train":
            data_len = self.train.shape[0]
            start_idx = _compute_window_start(index, data_len, self.win_size, self.step)
            data = np.float32(self.train[start_idx:start_idx + self.win_size])
            labels = np.float32(self.test_labels[0:self.win_size])
            return (data, labels)
        elif self.mode == 'test':
            data_len = self.test.shape[0]
            start_idx = _compute_window_start(index, data_len, self.win_size, self.step)
            data = np.float32(self.test[start_idx:start_idx + self.win_size])
            labels = np.float32(self.test_labels[start_idx:start_idx + self.win_size])
            return (data, labels, start_idx)
        else:
            data_len = self.train.shape[0]
            start_idx = _compute_window_start(index, data_len, self.win_size, self.step)
            data = np.float32(self.train[start_idx:start_idx + self.win_size])
            labels = np.float32(self.test_labels[0:self.win_size])
            return (data, labels)


class UCR_AnomalySegLoader(Dataset):
    """UCR Anomaly dataset loader with period estimation support."""

    def __init__(self, dataset_path, filename, win_size, step, mode="train", estimate_periods=False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.estimate_periods = estimate_periods

        fields = filename.split('_')
        if len(fields) < 5:
            raise ValueError(f"Invalid filename format: {filename}. "
                           f"Expected format: name_train_end_anomaly_start_anomaly_end.txt")

        train_end = int(fields[-3])
        anomaly_start = int(fields[-2]) - train_end
        anomaly_end_field = fields[-1]
        if anomaly_end_field.endswith('.txt'):
            anomaly_end_field = anomaly_end_field[:-4]
        anomaly_end = int(anomaly_end_field) - train_end

        file_path = os.path.join(dataset_path, filename)
        with open(file_path) as f:
            lines = f.readlines()
            if len(lines) == 1:
                values = [eval(val) for val in lines[0].strip().split(" ") if len(val) > 1]
            else:
                values = [eval(line.strip()) for line in lines]

        data = np.array(values).reshape(-1, 1)

        train_data = data[:train_end]
        scaler = StandardScaler()
        scaler.fit(train_data)
        data = scaler.transform(data).squeeze()

        self.train = data[:train_end].reshape(-1, 1)
        self.test = data[train_end:].reshape(-1, 1) if len(data[train_end:]) > 0 else np.array([]).reshape(-1, 1)

        self.test_labels = np.zeros(len(self.test))
        if len(self.test) > 0 and anomaly_start >= 0 and anomaly_end < len(self.test):
            self.test_labels[anomaly_start:anomaly_end + 1] = 1

        # Period estimation (UCR_Anomaly only)
        if self.estimate_periods:
            print(f"Estimating periods for UCR dataset: {filename}")
            estimator = PeriodEstimator()
            self.periods = estimator.estimate_periods_multivariate(self.train)
        else:
            self.periods = np.array([[0]])

        print("test:", self.test.shape)
        print("train:", self.train.shape)
        if self.estimate_periods:
            print("estimated periods:", self.periods.flatten())

    def get_periods(self):
        """Return estimated periods."""
        return self.periods

    def __len__(self):
        if self.mode == "train":
            return _compute_window_count(self.train.shape[0], self.win_size, self.step)
        elif self.mode == 'test':
            if len(self.test) < self.win_size:
                return 0
            return _compute_window_count(self.test.shape[0], self.win_size, self.step)
        else:
            return _compute_window_count(self.train.shape[0], self.win_size, self.step)

    def __getitem__(self, index):
        if self.mode == "train":
            data_len = self.train.shape[0]
            start_idx = _compute_window_start(index, data_len, self.win_size, self.step)
            data = np.float32(self.train[start_idx:start_idx + self.win_size])
            labels = np.float32(self.test_labels[0:self.win_size])
            return (data, labels)
        elif self.mode == 'test':
            data_len = self.test.shape[0]
            start_idx = _compute_window_start(index, data_len, self.win_size, self.step)
            data = np.float32(self.test[start_idx:start_idx + self.win_size])
            labels = np.float32(self.test_labels[start_idx:start_idx + self.win_size])
            return (data, labels, start_idx)
        else:
            data_len = self.train.shape[0]
            start_idx = _compute_window_start(index, data_len, self.win_size, self.step)
            data = np.float32(self.train[start_idx:start_idx + self.win_size])
            labels = np.float32(self.test_labels[0:self.win_size])
            return (data, labels)


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train',
                       dataset='KDD', val_ratio=0.2, filename=None, estimate_periods=False,
                       feature_indices=None, enable_fft=True, device='cuda:0',
                       seq_len=None, long_term_multiplier=None,
                       downsample_method='linear_interpolation'):

    if feature_indices is None:
        feature_indices = [0, 1, 2, 3, 4, 5, 6]

    # Display downsampling info
    if seq_len is not None and long_term_multiplier is not None:
        print(f"Batch-optimized loader: features={feature_indices}, FFT={enable_fft}, device={device}")
        print(f"Downsampling enabled: {win_size} -> {seq_len} (method={downsample_method})")
    else:
        print(f"Batch-optimized loader: features={feature_indices}, FFT={enable_fft}, device={device}")
        print(f"Downsampling disabled")

    # Create custom collate function
    collate_fn = create_batch_feature_collate_fn(
        feature_indices=feature_indices,
        enable_fft=enable_fft,
        device=device,
        seq_len=seq_len,
        long_term_multiplier=long_term_multiplier,
        downsample_method=downsample_method
    )

    # Create dataset
    periods = None

    if dataset == 'SMD':
        dataset_obj = SMDSegLoader(data_path, win_size, step, mode)
    elif dataset == 'MSL':
        dataset_obj = MSLSegLoader(data_path, win_size, step, mode)
    elif dataset == 'SMAP':
        dataset_obj = SMAPSegLoader(data_path, win_size, step, mode)
    elif dataset == 'PSM':
        dataset_obj = PSMSegLoader(data_path, win_size, step, mode)
    elif dataset == 'SWaT':
        dataset_obj = SWaTSegLoader(data_path, win_size, step, mode)
    elif dataset == 'UCR_Anomaly':
        dataset_obj = UCR_AnomalySegLoader(
            data_path, filename, win_size, step, mode, estimate_periods
        )
        periods = dataset_obj.get_periods()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset_len = len(dataset_obj)

        if dataset == 'UCR_Anomaly':
            min_windows_for_val = max(1, int(1 / val_ratio))
            if dataset_len < min_windows_for_val:
                print(f"Too few windows ({dataset_len}). Regenerating with step=1...")
                dataset_obj = UCR_AnomalySegLoader(
                    data_path, filename, win_size, step=1,
                    mode=mode, estimate_periods=estimate_periods
                )
                periods = dataset_obj.get_periods()
                dataset_len = len(dataset_obj)

            train_use_len = int(dataset_len * (1 - val_ratio))
            val_use_len = dataset_len - train_use_len
            indices = torch.arange(dataset_len, dtype=torch.long)
            train_subset = Subset(dataset_obj, indices[:train_use_len])
            val_subset = Subset(dataset_obj, indices[train_use_len:])

            k_use_len = max(int(train_use_len * 0.1), 10)
            k_use_len = min(k_use_len, train_use_len)
            k_sub_indices = indices[:k_use_len]
            k_subset = Subset(dataset_obj, k_sub_indices)
            print(f"train={train_use_len}, val={val_use_len}, k={k_use_len}")

        else:
            train_use_len = int(dataset_len * (1 - val_ratio))
            val_use_len = int(dataset_len * val_ratio)
            val_start_index = random.randrange(train_use_len)

            indices = torch.arange(dataset_len, dtype=torch.long)
            train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index + val_use_len:]])
            train_subset = Subset(dataset_obj, train_sub_indices)

            val_sub_indices = indices[val_start_index:val_start_index + val_use_len]
            val_subset = Subset(dataset_obj, val_sub_indices)

            k_use_len = max(int(train_use_len * 0.1), 10)
            k_sub_indices = indices[:k_use_len]
            k_subset = Subset(dataset_obj, k_sub_indices)
            print(f"train={train_use_len}, val={val_use_len}, k={k_use_len}")

        # Create DataLoaders
        train_loader = DataLoader(
            dataset=train_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            dataset=val_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=collate_fn
        )
        k_loader = DataLoader(
            dataset=k_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=collate_fn
        )

        return (train_loader, val_loader, k_loader), periods

    # Test DataLoader
    data_loader = DataLoader(
        dataset=dataset_obj,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn
    )

    return (data_loader, data_loader), periods