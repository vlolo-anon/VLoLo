import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer, CNNFeatureFusion
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, FeatureEmbedding_inverted
from models.short_term_memory_module import MemoryModule


class ShortTermModel(nn.Module):
    """Short-term model with memory-augmented reconstruction."""

    def __init__(self, seq_len, long_term_multiplier, d_model, input_c, output_c, n_features, n_heads, e_layers, d_ff,
                 factor, dropout, activation, output_attention, memory_initial,
                 n_memory, device, phase_type, dataset_name, ucr_filename=None,
                 shrink_thres=0.0025, memory_init_embedding=None, run_name=None, run_id=None):
        super(ShortTermModel, self).__init__()

        self.seq_len = seq_len
        self.long_seq_len = seq_len * long_term_multiplier
        self.output_attention = output_attention
        self.memory_initial = memory_initial

        # Embeddings
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, dropout)
        self.feature_embedding = FeatureEmbedding_inverted(
            seq_len=seq_len, n_features=n_features, d_model=d_model, dropout=dropout
        )

        # CNN Feature Fusion (alternative to Cross-Attention)
        self.cnn_feature_fusion = CNNFeatureFusion(d_model, n_features=n_features, dropout=dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Memory module
        self.mem_module = MemoryModule(
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

        # Projector for reconstruction
        self.projector = nn.Linear(2 * d_model, seq_len, bias=True)

    def forward(self, x, time_features):

        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, L, 1)

        _, _, N = x.shape

        # Feature Embedding: (B, N, F, L) -> (B, N, F, E)
        feature_embeddings = self.feature_embedding(time_features)

        # Series Embedding: (B, L, N) -> (B, N, E)
        series_embeddings = self.enc_embedding(x)

        # CNN Feature Fusion: (B, N, E), (B, N, F, E) -> (B, N, E)
        cross_attended_embeddings = self.cnn_feature_fusion(series_embeddings, feature_embeddings)

        # Encoder processing: (B, N, E) -> (B, N, E)
        encoded_embeddings, self_attns = self.encoder(cross_attended_embeddings, attn_mask=None)

        # Memory Module
        memory_outputs = self.mem_module(encoded_embeddings)
        memory_augmented = memory_outputs['output']  # (B, N, 2*E)
        memory_attn = memory_outputs['attn']  # (B, N, n_memory)
        memory_embeddings = memory_outputs['memory_init_embedding']  # (N, n_memory, E)
        mem = self.mem_module.mem  # (N, n_memory, E)

        if self.memory_initial:
            return {
                "out": memory_augmented,
                "memory_item_embedding": None,
                "queries": encoded_embeddings,
                "mem": mem,
                "attn": memory_attn
            }
        else:
            # Reconstruction
            reconstructed = self.projector(memory_augmented).permute(0, 2, 1)

            # Remove last dimension if univariate
            if reconstructed.shape[-1] == 1:
                reconstructed = reconstructed.squeeze(-1)  # (B, L)

            return {
                "out": reconstructed,
                "memory_item_embedding": memory_embeddings,
                "queries": encoded_embeddings,
                "mem": mem,
                "attn": memory_attn
            }