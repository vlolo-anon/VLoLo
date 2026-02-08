import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_inverted
from models.long_term_memory_module import LongTermMemoryModule


class LongTermModel(nn.Module):
    """Long-term model with memory-augmented reconstruction and upsampling."""

    def __init__(self, seq_len, long_term_multiplier, d_model, input_c, output_c, n_heads, e_layers, d_ff,
                 factor, dropout, activation, output_attention, memory_initial, n_memory, device, phase_type,
                 dataset_name, ucr_filename=None, shrink_thres=0.0025, memory_init_embedding=None,
                 run_name=None, run_id=None, downsample_method='linear_interpolation'):
        super(LongTermModel, self).__init__()

        self.seq_len = seq_len
        self.long_seq_len = seq_len * long_term_multiplier
        self.long_term_multiplier = long_term_multiplier
        self.output_attention = output_attention
        self.memory_initial = memory_initial
        self.downsample_method = downsample_method

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, dropout)

        # Linear encoder (alternative to Transformer)
        self.linear_encoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Long-term memory module
        self.long_term_mem_module = LongTermMemoryModule(
            n_memory=n_memory,
            fea_dim=d_model,
            n_vars=input_c,
            shrink_thres=shrink_thres,
            device=device,
            memory_init_embedding=memory_init_embedding,
            phase_type=phase_type,
            dataset_name=dataset_name,
            file_name=ucr_filename,
            run_name=run_name,
            run_id=run_id
        )

        # Projector
        self.projector = nn.Linear(2 * d_model, seq_len, bias=True)

    def upsample_time_series(self, data, target_length, method='linear_interpolation'):

        B, L_short, N = data.shape

        if method in ['linear_interpolation', 'spline_interpolation']:
            # PyTorch GPU interpolation
            data_reshaped = data.permute(0, 2, 1)  # (B, N, L_short)
            upsampled_reshaped = F.interpolate(
                data_reshaped,
                size=target_length,
                mode='linear',
                align_corners=True
            )
            upsampled = upsampled_reshaped.permute(0, 2, 1)  # (B, L_long, N)
            return upsampled

        elif method == 'nearest':
            data_reshaped = data.permute(0, 2, 1)
            upsampled_reshaped = F.interpolate(
                data_reshaped,
                size=target_length,
                mode='nearest'
            )
            upsampled = upsampled_reshaped.permute(0, 2, 1)
            return upsampled

        elif method == 'repeat':
            repeat_factor = target_length // L_short
            remainder = target_length % L_short

            repeated = data.repeat_interleave(repeat_factor, dim=1)

            if remainder > 0:
                extra = data[:, :remainder, :]
                repeated = torch.cat([repeated, extra], dim=1)

            return repeated

        else:
            return self.upsample_time_series(data, target_length, 'linear_interpolation')

    def forward(self, downsampled_x):
        """
        Args:
            downsampled_x: (B, L_short, N) downsampled long-term time series

        Returns:
            Output dictionary with reconstruction and memory information
        """
        _, _, N = downsampled_x.shape

        # Series Embedding: (B, L_short, N) -> (B, N, E)
        series_embedding = self.enc_embedding(downsampled_x)

        # Linear encoder: (B, N, E) -> (B, N, E)
        combined_queries = self.linear_encoder(series_embedding)

        # Long-term memory module
        memory_outputs = self.long_term_mem_module(combined_queries)
        memory_augmented = memory_outputs['output']  # (B, N, 2*E)
        memory_attn = memory_outputs['attn']  # (B, N, n_memory)
        memory_embeddings = memory_outputs['memory_init_embedding']  # (N, n_memory, E)
        mem = self.long_term_mem_module.mem  # (N, n_memory, E)

        if self.memory_initial:
            return {
                "out": memory_augmented,
                "memory_item_embedding": None,
                "queries": combined_queries,
                "mem": mem,
                "attn": memory_attn
            }
        else:
            # Projector: (B, N, 2*E) -> (B, N, L_short)
            reconstructed_short = self.projector(memory_augmented)
            reconstructed_short = reconstructed_short.permute(0, 2, 1)  # (B, L_short, N)

            # Upsample to original long-term length
            reconstructed_long = self.upsample_time_series(
                reconstructed_short,
                target_length=self.long_seq_len,
                method=getattr(self, 'upsample_method', 'linear_interpolation')
            )

            # Remove last dimension if univariate
            if reconstructed_long.shape[-1] == 1:
                reconstructed_long = reconstructed_long.squeeze(-1)

            return {
                "out": reconstructed_long,
                "out_short": reconstructed_short,
                "memory_item_embedding": memory_embeddings,
                "queries": combined_queries,
                "mem": mem,
                "attn": memory_attn
            }