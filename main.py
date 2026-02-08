import os
import argparse
import torch
import glob
import numpy as np

from torch.backends import cudnn
from utils.utils import *
from solver import Solver


def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(config):
    """Main function for training and evaluation."""
    cudnn.benchmark = True

    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path, exist_ok=True)

    METRIC_KEYS = [
        'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR',
    ]

    if config.dataset == 'UCR_Anomaly':
        # Determine file list
        if config.ucr_filename:
            # Process specified file only
            files = [os.path.join(config.data_path, config.ucr_filename)]
            if not os.path.exists(files[0]):
                print(f"Error: File not found: {files[0]}")
                return None
        else:
            # Process all UCR_Anomaly files
            pattern = os.path.join(config.data_path, "*.txt")
            files = sorted(glob.glob(pattern))

        for file_path in files:
            filename = os.path.basename(file_path)
            print(f"\n{'=' * 80}")
            print(f"Processing file: {filename}")
            print(f"{'=' * 80}")

            file_config = vars(config).copy()
            file_config['ucr_filename'] = filename

            file_results = []

            for run_id in range(config.k):
                print(f"\n{'-' * 60}")
                print(f"Run {run_id + 1}/{config.k} for {filename}")
                print(f"{'-' * 60}")

                file_config['run_id'] = run_id + 1

                solver = Solver(file_config)

                if config.mode == 'train_first_step':
                    solver.first_step_train_both_integrated()
                elif config.mode == 'train_second_step':
                    success = solver.second_step_get_memory_embeddings()
                    if success:
                        solver.second_step_cooperative_train()
                        solver.save_memory_items()
                    else:
                        print(f"Skipping run {run_id + 1} due to missing checkpoint")
                        continue
                elif config.mode == 'test':
                    result = solver.test()
                    if result is None:
                        continue
                    metrics = tuple(result[key] for key in METRIC_KEYS)
                    file_results.append(metrics)

            # Display average results for this file
            if config.mode == 'test' and len(file_results) > 0:
                avg_results = np.mean(file_results, axis=0)
                std_results = np.std(file_results, axis=0) if len(file_results) > 1 else np.zeros(len(METRIC_KEYS))

                print(f"\n{'*' * 60}")
                print(f"RESULTS FOR {filename} ({config.k} runs)")
                print(f"{'*' * 60}")
                for i, key in enumerate(METRIC_KEYS):
                    print(f"{key:<22}: {avg_results[i]:.3f} ± {std_results[i]:.4f}")
                print(f"{'*' * 60}")

    else:
        # Process other datasets
        results = []

        for run_id in range(config.k):
            print(f"\n{'=' * 60}")
            print(f"Run {run_id + 1}/{config.k}")
            print(f"{'=' * 60}")

            config_dict = vars(config).copy()
            config_dict['run_id'] = run_id + 1

            solver = Solver(config_dict)

            if config.mode == 'train_first_step':
                solver.first_step_train_both_integrated()
            elif config.mode == 'train_second_step':
                success = solver.second_step_get_memory_embeddings()
                if success:
                    solver.second_step_cooperative_train()
                    solver.save_memory_items()
                else:
                    print(f"Skipping run {run_id + 1} due to missing checkpoint")
                    continue
            elif config.mode == 'test':
                result = solver.test()
                if result is None:
                    continue
                metrics = tuple(result[key] for key in METRIC_KEYS)
                results.append(metrics)

        # Display overall average results
        if config.mode == 'test' and len(results) > 0:
            avg_results = np.mean(results, axis=0)
            std_results = np.std(results, axis=0) if len(results) > 1 else np.zeros(len(METRIC_KEYS))

            print(f"\n{'=' * 60}")
            print("OVERALL AVERAGE RESULTS")
            print(f"{'=' * 60}")
            for i, key in enumerate(METRIC_KEYS):
                print(f"{key:<22}: {avg_results[i]:.3f} ± {std_results[i]:.4f}")
            print(f"{'=' * 60}")

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument('--lr', type=float, default=0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--k', type=int, default=1, help='Number of runs')
    parser.add_argument('--run_name', type=str, default='run1', help='Experiment name for result folder')
    parser.add_argument('--test_name', type=str, default='test', help='Test name for result folder')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lambd', type=float, default=0.01, help='Lambda for entropy loss')

    # Data parameters
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=51)
    parser.add_argument('--output_c', type=int, default=51)
    parser.add_argument('--dataset', type=str, default='SWaT')
    parser.add_argument('--data_path', type=str, default='../datasets/SWaT/')

    # Mode and device
    parser.add_argument('--mode', type=str, default='test',
                        choices=['train_first_step', 'train_second_step', 'test'])
    parser.add_argument('--model_save_path', type=str, default='save_models/VLoLo')
    parser.add_argument('--device', type=str, default="cuda:0")

    # Memory parameters
    parser.add_argument('--n_memory', type=int, default=10, help='Number of memory items for short-term model')
    parser.add_argument('--n_memory_long', type=int, default=5, help='Number of memory items for long-term model')
    parser.add_argument('--memory_initial', type=str2bool, default=False,
                        help='Whether to use random initialization (True) or K-means initialization (False)')

    # Model architecture
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Dimension of feed-forward network')
    parser.add_argument('--factor', type=int, default=1, help='Attention factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='Time features encoding: timeF, fixed, learned')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
    parser.add_argument('--output_attention', action='store_true', help='Whether to output attention in encoder')
    parser.add_argument('--phase_type', type=str, default=None, help='Phase type for period encoding')

    # Hierarchical model parameters
    parser.add_argument('--alpha_mode', type=str, default='fixed',
                        choices=['fixed', 'learnable'],
                        help='Alpha blending mode: fixed value or learnable parameter')
    parser.add_argument('--alpha_value', type=float, default=0.5,
                        help='Fixed alpha value (used when alpha_mode=fixed)')
    parser.add_argument('--long_term_multiplier', type=int, default=5, help='Long-term sequence multiplier')
    parser.add_argument('--downsample_method', type=str, default='linear_interpolation',
                        choices=['mean', 'max', 'min', 'median', 'first', 'last',
                                 'linear_interpolation', 'spline_interpolation', 'peak_detection',
                                 'perceptually_important', 'adaptive_density', 'gradient_change', 'variance_based'],
                        help='Downsampling method for long-term model')
    parser.add_argument('--upsample_method', type=str, default='linear_interpolation',
                        choices=['linear_interpolation', 'spline_interpolation', 'repeat', 'nearest'],
                        help='Upsampling method for long-term model reconstruction')

    # Feature parameters
    parser.add_argument('--feature_indices', type=int, nargs='*',
                        help='Feature indices to use (0-6). Example: --feature_indices 0 1 2 4')

    # UCR_Anomaly specific
    parser.add_argument('--ucr_filename', type=str, default=None,
                        help='Specific UCR filename to process (if None, process all files)')
    parser.add_argument('--use_score_smoothing', action='store_true', default=False,
                        help='Apply moving average smoothing to anomaly scores (UCR_Anomaly only)')

    config = parser.parse_args()
    args = vars(config)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    main(config)
    print("Program completed successfully!")
    os._exit(0)