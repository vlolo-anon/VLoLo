import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm

from utils.utils import *
from utils.long_term_utils import *
from models.hierarchical_model import HierarchicalModel
from models.loss_functions import *
from data_factory.data_loader import get_loader_segment
from sklearn.preprocessing import MinMaxScaler
from vus.metrics import get_metrics
from sklearn.metrics import precision_recall_fscore_support
import logging


def adjust_learning_rate(optimizer, epoch, lr_):
    """Adjust learning rate with exponential decay."""
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class OneEarlyStopping:
    """Early stopping handler to prevent overfitting."""

    def __init__(self, patience=10, verbose=False, dataset_name='', delta=0, type=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.dataset = dataset_name
        self.type = type

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + f'_checkpoint_{self.type}.pth'))
        self.val_loss_min = val_loss


class Solver(object):
    """Main solver class for training and evaluation."""

    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        # Period information (only for UCR_Anomaly)
        self.dataset_periods = None

        # Score smoothing option (for UCR_Anomaly)
        self.use_score_smoothing = getattr(self, 'use_score_smoothing', False)

        # Setup data loaders
        self._setup_data_loaders()

        # Setup device
        self._setup_device()

        # Initialize memory embeddings
        self.memory_init_embedding = None
        self.long_term_memory_init_embedding = None

        # Build model
        self.build_model()

        # Loss functions
        self.entropy_loss = EntropyLoss()
        self.criterion = nn.MSELoss()

        # Setup logger
        self._setup_logger()

    def _setup_data_loaders(self):
        """Setup data loaders with long-term window size."""
        long_win_size = self.seq_len * self.long_term_multiplier

        if hasattr(self, 'ucr_filename') and self.ucr_filename is not None:
            # UCR-Anomaly dataset (with period estimation)
            loaders, self.dataset_periods = get_loader_segment(
                self.data_path,
                batch_size=self.batch_size,
                win_size=long_win_size,
                step=long_win_size,
                mode='train',
                dataset=self.dataset,
                filename=self.ucr_filename,
                estimate_periods=True,
                feature_indices=self.feature_indices,
                seq_len=self.seq_len,
                long_term_multiplier=self.long_term_multiplier
            )
            self.train_loader, self.vali_loader, self.k_loader = loaders

            test_loaders, _ = get_loader_segment(
                self.data_path,
                batch_size=self.batch_size,
                win_size=long_win_size,
                step=long_win_size,
                mode='test',
                dataset=self.dataset,
                filename=self.ucr_filename,
                estimate_periods=False,
                feature_indices=self.feature_indices,
                seq_len=self.seq_len,
                long_term_multiplier=self.long_term_multiplier
            )
            self.test_loader, _ = test_loaders
        else:
            # Other datasets (no period estimation)
            loaders, _ = get_loader_segment(
                self.data_path,
                batch_size=self.batch_size,
                win_size=long_win_size,
                step=long_win_size,
                mode='train',
                dataset=self.dataset,
                estimate_periods=False,
                feature_indices=self.feature_indices,
                seq_len=self.seq_len,
                long_term_multiplier=self.long_term_multiplier
            )
            self.train_loader, self.vali_loader, self.k_loader = loaders

            test_loaders, _ = get_loader_segment(
                self.data_path,
                batch_size=self.batch_size,
                win_size=long_win_size,
                step=long_win_size,
                mode='test',
                dataset=self.dataset,
                estimate_periods=False,
                feature_indices=self.feature_indices,
                seq_len=self.seq_len,
                long_term_multiplier=self.long_term_multiplier
            )
            self.test_loader, _ = test_loaders

        print(f"Data loaders configured for long-term size: {long_win_size}")
        print(f"Short-term equivalent size: {self.seq_len}")
        print(f"Long-term multiplier: {self.long_term_multiplier}")
        print(f"Training batches: {len(self.train_loader)}")

        # Display period information (UCR_Anomaly only)
        if self.dataset_periods is not None:
            period_value = self.dataset_periods[0, 0]
            if period_value > 0:
                print(f"Estimated period: {period_value}")

    def _setup_device(self):
        """Setup computation device (GPU/CPU)."""
        device_name = getattr(self, 'device', 'cuda:0')

        if torch.cuda.is_available():
            try:
                device_id = int(device_name.split(':')[1]) if ':' in device_name else 0
                if device_id < torch.cuda.device_count():
                    self.device = torch.device(device_name)
                else:
                    print(f"Warning: GPU {device_name} not available, using cuda:0")
                    self.device = torch.device("cuda:0")
            except:
                print(f"Warning: Invalid device {device_name}, using cuda:0")
                self.device = torch.device("cuda:0")
        else:
            print("CUDA not available, using CPU")
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

    def _setup_logger(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def _get_clean_filename(self):
        """Get clean filename without extension for UCR dataset."""
        if hasattr(self, 'ucr_filename') and self.ucr_filename is not None:
            return os.path.splitext(self.ucr_filename)[0]
        return None

    def _get_save_path(self):
        """Get model checkpoint save path."""
        if self.dataset == 'UCR_Anomaly' and hasattr(self, 'ucr_filename') and self.ucr_filename is not None:
            clean_filename = self._get_clean_filename()
            save_path = os.path.join(
                self.model_save_path, self.run_name, str(self.run_id),
                'checkpoints', self.dataset, clean_filename
            )
        else:
            save_path = os.path.join(
                self.model_save_path, self.run_name, str(self.run_id), 'checkpoints'
            )

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        return save_path

    def _reshape_features_for_short_term(self, input_long, feature_long):

        batch_size = input_long.shape[0]

        if len(input_long.shape) == 3:  # Multivariate: (B, L, C)
            long_seq_len = input_long.shape[1]
            channels = input_long.shape[2]
            input_short_reshaped = input_long.reshape(
                batch_size * self.long_term_multiplier, self.seq_len, channels
            )
        else:  # Univariate: (B, L)
            long_seq_len = input_long.shape[1]
            channels = None
            input_short_reshaped = input_long.reshape(
                batch_size * self.long_term_multiplier, self.seq_len
            )

        # Feature shape transformation
        B, N, F, total_len = feature_long.shape
        feature_short_reshaped = (
            feature_long
            .reshape(B, N, F, self.long_term_multiplier, self.seq_len)
            .permute(0, 3, 1, 2, 4)  # (B, multiplier, N, F, seq_len)
            .reshape(B * self.long_term_multiplier, N, F, self.seq_len)
        )

        return input_short_reshaped, feature_short_reshaped, channels, long_seq_len

    def _reshape_recon_to_long(self, short_recon, batch_size, long_seq_len, channels):

        if channels is not None:
            return short_recon.reshape(batch_size, long_seq_len, channels)
        else:
            return short_recon.reshape(batch_size, long_seq_len)

    def _get_eval_dir(self):
        """Get evaluation results directory."""
        if self.dataset == 'UCR_Anomaly' and hasattr(self, 'ucr_filename') and self.ucr_filename is not None:
            clean_filename = self._get_clean_filename()
            eval_dir = os.path.join(
                f"save_models/VLoLo/{self.run_name}/{self.run_id}/evaluation_results/{self.test_name}",
                self.dataset, clean_filename
            )
        else:
            eval_dir = os.path.join(
                f"save_models/VLoLo/{self.run_name}/{self.run_id}/evaluation_results/{self.test_name}",
                self.dataset
            )
        os.makedirs(eval_dir, exist_ok=True)
        return eval_dir

    def build_model(self, memory_init_embedding=None):
        """Build hierarchical model."""
        if memory_init_embedding is None:
            memory_init_embedding = self.memory_init_embedding

        long_term_memory_init_embedding = getattr(self, 'long_term_memory_init_embedding', None)

        self.model = HierarchicalModel(
            seq_len=self.seq_len,
            long_term_multiplier=self.long_term_multiplier,
            d_model=self.d_model,
            input_c=self.input_c,
            output_c=self.output_c,
            n_features=len(self.feature_indices),
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            factor=self.factor,
            dropout=self.dropout,
            activation=self.activation,
            output_attention=self.output_attention,
            memory_initial=self.memory_initial,
            n_memory=self.n_memory,
            n_memory_long=self.n_memory_long,
            device=self.device,
            phase_type=self.phase_type,
            dataset_name=self.dataset,
            ucr_filename=getattr(self, 'ucr_filename', None),
            shrink_thres=getattr(self, 'shrink_thres', 0.0025),
            memory_init_embedding=memory_init_embedding,
            long_term_memory_init_embedding=long_term_memory_init_embedding,
            run_name=self.run_name,
            run_id=self.run_id,
            downsample_method=getattr(self, 'downsample_method', 'linear_interpolation'),
            upsample_method=getattr(self, 'upsample_method', 'linear_interpolation'),
            alpha_mode=getattr(self, 'alpha_mode', 'fixed'),
            alpha_value=getattr(self, 'alpha_value', 0.5)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def first_step_train_both_integrated(self):
        """First step: Integrated training of short-term and long-term models."""
        print("=" * 22 + "FIRST STEP - INTEGRATED TRAINING" + "=" * 22)

        early_stopping = OneEarlyStopping(
            patience=10, verbose=True, dataset_name=self.dataset, type='first_integrated'
        )
        path = self._get_save_path()

        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()
            loss_list = []

            for i, (input_long, labels, feature_long, downsampled_input, start_indices) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                input_long = input_long.float().to(self.device)
                if input_long.shape[-1] == 1:
                    input_long = input_long.squeeze(-1)

                batch_size = input_long.shape[0]

                # Reshape for short-term model
                input_short_reshaped, feature_short_reshaped, channels, long_seq_len = \
                    self._reshape_features_for_short_term(input_long, feature_long)

                # Short-term reconstruction
                short_output = self.model(
                    x_short=input_short_reshaped,
                    model_type='short',
                    feature_short_reshaped=feature_short_reshaped
                )
                short_recon = short_output['out']

                # Reshape to long-term size
                short_recon_long = self._reshape_recon_to_long(
                    short_recon, batch_size, long_seq_len, channels
                )

                # Long-term reconstruction
                long_output = self.model(x_long=downsampled_input, model_type='long')
                long_recon = long_output['out']

                # Integration
                alpha = self.model.get_alpha()
                integrated_recon = alpha * short_recon_long + (1 - alpha) * long_recon

                # Loss calculation
                rec_loss = self.criterion(integrated_recon, input_long)

                entropy_loss_short = self.entropy_loss(short_output['attn'])
                entropy_loss_long = self.entropy_loss(long_output['attn'])

                alpha_item = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
                entropy_loss = alpha_item * entropy_loss_short + (1 - alpha_item) * entropy_loss_long
                loss = rec_loss + self.lambd * entropy_loss

                loss_list.append(loss.item())

                # Parameter update
                loss.backward()
                self.optimizer.step()

            train_loss = np.average(loss_list) if loss_list else 0.0
            valid_loss = self.vali_integrated()

            alpha_display = self.model.get_alpha()
            alpha_value = alpha_display.item() if isinstance(alpha_display, torch.Tensor) else alpha_display
            print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} | Valid Loss: {valid_loss:.7f}")
            print(f"Alpha value: {alpha_value:.4f}")

            early_stopping(valid_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping in first step integrated training")
                break

    def second_step_get_memory_embeddings(self):
        """Second step: Get memory embeddings via K-means clustering."""
        print("=" * 22 + "GETTING MEMORY EMBEDDINGS" + "=" * 22)

        # Load first step checkpoint
        path = self._get_save_path()
        first_checkpoint = os.path.join(path, f"{self.dataset}_checkpoint_first_integrated.pth")

        if not os.path.exists(first_checkpoint):
            print(f"ERROR: First step checkpoint not found: {first_checkpoint}")
            print("Please run First Step training first.")
            return False

        print(f"Loading first step model from: {first_checkpoint}")
        self.model.load_state_dict(torch.load(first_checkpoint))
        self.model.eval()

        # Collect queries for memory initialization
        short_queries = []
        long_queries = []

        for i, (input_long, labels, feature_long, downsampled_input, start_indices) in enumerate(self.k_loader):
            input_long = input_long.float().to(self.device)
            if input_long.shape[-1] == 1:
                input_long = input_long.squeeze(-1)

            # Get short-term queries
            input_short_reshaped, feature_short_reshaped, channels, long_seq_len = \
                self._reshape_features_for_short_term(input_long, feature_long)

            short_output = self.model(
                x_short=input_short_reshaped,
                model_type='short',
                feature_short_reshaped=feature_short_reshaped
            )
            short_queries.append(short_output['queries'])

            # Get long-term queries
            long_output = self.model(x_long=downsampled_input, model_type='long')
            long_queries.append(long_output['queries'])

        # Concatenate queries
        short_combined_queries = torch.cat(short_queries, dim=0)
        long_combined_queries = torch.cat(long_queries, dim=0)

        # K-means clustering for short-term memory
        print("Running K-means for short-term memory...")
        short_memory_result = fixed_k_means_clustering(
            x=short_combined_queries, n_mem=self.n_memory, d_model=self.d_model
        )
        self.memory_init_embedding = short_memory_result.squeeze(0).detach()

        # K-means clustering for long-term memory
        print("Running K-means for long-term memory...")
        long_memory_result = fixed_long_term_k_means_clustering(
            x=long_combined_queries, n_mem=self.n_memory_long, d_model=self.d_model
        )
        self.long_term_memory_init_embedding = long_memory_result.squeeze(0).detach()

        print('Memory shapes:', self.memory_init_embedding.shape, self.long_term_memory_init_embedding.shape)

        # Rebuild model with initialized memory
        self.memory_initial = False
        self.build_model()

        return True

    def second_step_cooperative_train(self):
        """Second step: Cooperative training with initialized memory."""
        print("=" * 22 + "SECOND STEP - COOPERATIVE TRAINING" + "=" * 22)

        early_stopping = OneEarlyStopping(
            patience=10, verbose=True, dataset_name=self.dataset, type='integrated'
        )
        save_path = self._get_save_path()

        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()
            loss_list = []

            for i, (input_long, labels, feature_long, downsampled_input, start_indices) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                input_long = input_long.float().to(self.device)
                if input_long.shape[-1] == 1:
                    input_long = input_long.squeeze(-1)

                batch_size = input_long.shape[0]

                # Reshape for short-term model
                input_short_reshaped, feature_short_reshaped, channels, long_seq_len = \
                    self._reshape_features_for_short_term(input_long, feature_long)

                # Short-term reconstruction
                short_output = self.model(
                    x_short=input_short_reshaped,
                    model_type='short',
                    feature_short_reshaped=feature_short_reshaped
                )
                short_recon = short_output['out']

                # Reshape to long-term size
                short_recon_long = self._reshape_recon_to_long(
                    short_recon, batch_size, long_seq_len, channels
                )

                # Long-term reconstruction
                long_output = self.model(x_long=downsampled_input, model_type='long')
                long_recon = long_output['out']

                # Integration
                alpha = self.model.get_alpha()
                integrated_recon = alpha * short_recon_long + (1 - alpha) * long_recon

                # Loss calculation
                rec_loss = self.criterion(integrated_recon, input_long)

                entropy_loss_short = self.entropy_loss(short_output['attn'])
                entropy_loss_long = self.entropy_loss(long_output['attn'])

                alpha_item = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
                entropy_loss = alpha_item * entropy_loss_short + (1 - alpha_item) * entropy_loss_long
                loss = rec_loss + self.lambd * entropy_loss

                loss_list.append(loss.item())

                loss.backward()
                self.optimizer.step()

            train_loss = np.average(loss_list) if loss_list else 0.0
            valid_loss = self.vali_integrated()

            alpha_display = self.model.get_alpha()
            alpha_value = alpha_display.item() if isinstance(alpha_display, torch.Tensor) else alpha_display
            print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} | Valid Loss: {valid_loss:.7f}")
            print(f"Alpha value: {alpha_value:.4f}")

            early_stopping(valid_loss, self.model, save_path)
            if early_stopping.early_stop:
                print("Early stopping in second step cooperative training")
                break

    def vali_integrated(self):
        """Validation for integrated training."""
        self.model.eval()
        valid_loss_list = []

        with torch.no_grad():
            for i, (input_long, labels, feature_long, downsampled_input, start_indices) in enumerate(self.vali_loader):
                input_long = input_long.float().to(self.device)
                if input_long.shape[-1] == 1:
                    input_long = input_long.squeeze(-1)

                batch_size = input_long.shape[0]

                # Reshape for short-term model
                input_short_reshaped, feature_short_reshaped, channels, long_seq_len = \
                    self._reshape_features_for_short_term(input_long, feature_long)

                # Short-term reconstruction
                short_output = self.model(
                    x_short=input_short_reshaped,
                    model_type='short',
                    feature_short_reshaped=feature_short_reshaped
                )
                short_recon = short_output['out']

                # Reshape to long-term size
                short_recon_long = self._reshape_recon_to_long(
                    short_recon, batch_size, long_seq_len, channels
                )

                # Long-term reconstruction
                long_output = self.model(x_long=downsampled_input, model_type='long')
                long_recon = long_output['out']

                # Integration
                alpha = self.model.get_alpha()
                integrated_recon = alpha * short_recon_long + (1 - alpha) * long_recon

                # Loss calculation
                rec_loss = self.criterion(integrated_recon, input_long)

                entropy_loss_short = self.entropy_loss(short_output['attn'])
                entropy_loss_long = self.entropy_loss(long_output['attn'])

                alpha_item = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
                entropy_loss = alpha_item * entropy_loss_short + (1 - alpha_item) * entropy_loss_long
                loss = rec_loss + self.lambd * entropy_loss

                valid_loss_list.append(loss.item())

        return np.average(valid_loss_list) if valid_loss_list else 0.0

    def test(self):
        """Test the integrated model and compute anomaly scores."""
        save_path = self._get_save_path()
        checkpoint_path = os.path.join(save_path, f"{self.dataset}_checkpoint_integrated.pth")

        print(f"Loading integrated model from: {checkpoint_path}")
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.to(self.device)
        self.model.eval()
        print("=" * 22 + "INTEGRATED TEST MODE" + "=" * 22)

        criterion = nn.MSELoss(reduction='none')

        # Get test data length from dataset
        test_dataset = self.test_loader.dataset
        if hasattr(test_dataset, 'dataset'):
            # Subset case
            base_dataset = test_dataset.dataset
        else:
            base_dataset = test_dataset

        test_data_len = len(base_dataset.test)
        long_win_size = self.seq_len * self.long_term_multiplier

        print(f"Test data length: {test_data_len}, Window size: {long_win_size}")

        # Initialize score arrays
        test_energy = np.zeros(test_data_len)
        test_labels = np.zeros(test_data_len)
        evaluated = np.zeros(test_data_len, dtype=bool)

        reconstructed_output = []
        original_output = []

        with torch.no_grad():
            for i, (input_long, labels, feature_long, downsampled_input, start_indices) in enumerate(self.test_loader):
                input_long = input_long.float().to(self.device)

                if input_long.shape[-1] == 1:
                    input_long = input_long.squeeze(-1)

                batch_size = input_long.shape[0]

                # Reshape for short-term model
                input_short_reshaped, feature_short_reshaped, channels, long_seq_len = \
                    self._reshape_features_for_short_term(input_long, feature_long)

                # Short-term reconstruction
                short_output = self.model(
                    x_short=input_short_reshaped,
                    model_type='short',
                    feature_short_reshaped=feature_short_reshaped
                )
                short_recon = short_output['out']

                # Reshape to long-term size
                short_recon_long = self._reshape_recon_to_long(
                    short_recon, batch_size, long_seq_len, channels
                )

                # Long-term reconstruction
                long_output = self.model(x_long=downsampled_input, model_type='long')
                long_recon = long_output['out']

                # Integration
                alpha = self.model.get_alpha()
                integrated_recon = alpha * short_recon_long + (1 - alpha) * long_recon

                # Handle univariate case
                if self.input_c == 1 and integrated_recon.dim() == 2:
                    integrated_recon = integrated_recon.unsqueeze(-1)
                    input_long_for_loss = input_long.unsqueeze(-1) if input_long.dim() == 2 else input_long
                else:
                    input_long_for_loss = input_long

                # Compute reconstruction error: (B, L, C) -> (B, L) by mean over channels
                rec_loss = torch.mean(criterion(input_long_for_loss, integrated_recon), dim=-1)
                scores = rec_loss.detach().cpu().numpy()  # (B, L)
                labels_np = labels.cpu().numpy()  # (B, L) or (B, L, 1)

                if labels_np.ndim == 3:
                    labels_np = labels_np.squeeze(-1)

                # Aggregate scores: only add non-overlapping (unevaluated) points
                start_indices_np = start_indices.cpu().numpy()
                for b in range(batch_size):
                    start = start_indices_np[b]
                    end = min(start + long_win_size, test_data_len)

                    # Mask for unevaluated points
                    mask = ~evaluated[start:end]
                    valid_len = end - start

                    # Only set scores for unevaluated points
                    indices = np.arange(start, end)
                    test_energy[indices[mask]] = scores[b, :valid_len][mask]
                    test_labels[indices[mask]] = labels_np[b, :valid_len][mask]
                    evaluated[start:end] = True

                reconstructed_output.append(integrated_recon.detach().cpu().numpy())
                original_output.append(input_long_for_loss.detach().cpu().numpy())

        reconstructed_output = np.concatenate(reconstructed_output, axis=0)
        original_output = np.concatenate(original_output, axis=0)

        reconstructed_output = reconstructed_output.reshape(-1, self.input_c)
        original_output = original_output.reshape(-1, self.input_c)

        # Display alpha value
        alpha_display = self.model.get_alpha()
        alpha_value = alpha_display.item() if isinstance(alpha_display, torch.Tensor) else alpha_display
        print(f'Final alpha value: {alpha_value:.4f}')
        print('test_energy:', test_energy.shape, 'test_labels:', test_labels.shape)
        print('original_output:', original_output.shape, 'reconstructed_output:', reconstructed_output.shape)

        results = self._evaluate_anomaly_detection(
            test_energy, test_labels, original_output, reconstructed_output
        )

        return results

    def convert_numpy_types(self, data):
        """Convert NumPy types to native Python types for JSON serialization."""
        if isinstance(data, dict):
            return {key: self.convert_numpy_types(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.convert_numpy_types(element) for element in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.floating, np.complexfloating)):
            return float(data)
        elif isinstance(data, (np.integer, np.signedinteger, np.unsignedinteger)):
            return int(data)
        elif hasattr(data, 'item'):
            return data.item()
        return data


    def _evaluate_anomaly_detection(self, scores, ground_truth, original_output, reconstructed_output):
        """Evaluate anomaly detection performance."""
        # Setup evaluation directory
        eval_dir = self._get_eval_dir()
        print(f"Saving evaluation results to: {eval_dir}")

        print(scores.flatten().shape, ground_truth.flatten().shape)

        # Normalize scores
        scores = MinMaxScaler((0, 1)).fit_transform(scores.reshape(-1, 1)).ravel()

        # Score smoothing for UCR_Anomaly (using estimated period)
        if self.dataset == 'UCR_Anomaly' and self.use_score_smoothing and self.dataset_periods is not None:
            period_value = int(self.dataset_periods[0, 0])
            if period_value > 0:
                window_size = period_value
                print(f"Applying score smoothing with window size: {window_size}")
                scores = pd.Series(scores)
                scores = scores.rolling(window=window_size, center=True).mean().fillna(scores).to_numpy()

        ground_truth_flat = ground_truth.flatten().astype(int)

        if ground_truth_flat.sum() == 0:
            return None

        # Get VUS metrics
        results = get_metrics(
            scores, ground_truth_flat, metric='all', slidingWindow=self.seq_len
        )

        # Add smoothing info if used
        if self.dataset == 'UCR_Anomaly' and self.use_score_smoothing and self.dataset_periods is not None:
            period_value = int(self.dataset_periods[0, 0])
            if period_value > 0:
                results['smoothing_window_size'] = period_value

        # Display results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        for metric in results.keys():
            print(f"{metric:<22}: {results[metric]}")
        print("=" * 60)

        # Save metrics to JSON
        json_filename = "metrics.json"
        json_filepath = os.path.join(eval_dir, json_filename)

        json_results = self.convert_numpy_types(results)

        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=4, ensure_ascii=False)

        print(f"Metrics saved to: {json_filepath}")

        # Save data arrays
        np.save(os.path.join(eval_dir, "scores.npy"), scores)
        np.save(os.path.join(eval_dir, "ground_truth.npy"), ground_truth)
        np.save(os.path.join(eval_dir, "original_output.npy"), original_output)
        np.save(os.path.join(eval_dir, "reconstructed_output.npy"), reconstructed_output)

        print(f"Data arrays saved to: {eval_dir}")

        return results

    def save_memory_items(self):
        """Save memory items to disk."""
        print("=" * 22 + "SAVING MEMORY ITEMS" + "=" * 22)

        # Get short-term memory
        short_memory = self.model.short_term_model.mem_module.mem

        # Get long-term memory
        long_memory = self.model.long_term_model.long_term_mem_module.mem

        print('Save memory:', short_memory.shape, long_memory.shape)

        # Setup save directory
        if self.dataset == 'UCR_Anomaly' and hasattr(self, 'ucr_filename') and self.ucr_filename is not None:
            clean_filename = self._get_clean_filename()
            item_folder_path = os.path.join(
                f"save_models/VLoLo/{self.run_name}/{self.run_id}/memory_item",
                self.dataset, clean_filename
            )
        else:
            item_folder_path = f"save_models/VLoLo/{self.run_name}/{self.run_id}/memory_item"

        if not os.path.exists(item_folder_path):
            os.makedirs(item_folder_path)

        # Save short-term memory
        short_memory_path = os.path.join(item_folder_path, f"{self.dataset}_memory_item.pth")
        torch.save(short_memory, short_memory_path)
        print(f"Short-term memory saved to: {short_memory_path}")

        # Save long-term memory
        long_memory_path = os.path.join(item_folder_path, f"{self.dataset}_long_term_memory_item.pth")
        torch.save(long_memory, long_memory_path)
        print(f"Long-term memory saved to: {long_memory_path}")