import torch
import torch.nn as nn
from models.short_term_model import ShortTermModel
from models.long_term_model import LongTermModel


class HierarchicalModel(nn.Module):
    """Hierarchical model combining short-term and long-term models."""

    def __init__(self, seq_len, long_term_multiplier, d_model, input_c, output_c, n_features,
                 n_heads, e_layers, d_ff, factor, dropout, activation,
                 output_attention, memory_initial, n_memory, n_memory_long, device, phase_type,
                 dataset_name, ucr_filename, shrink_thres, memory_init_embedding, long_term_memory_init_embedding,
                 run_name, run_id, downsample_method='linear_interpolation',
                 upsample_method='linear_interpolation',
                 alpha_mode='fixed', alpha_value=0.5):
        super(HierarchicalModel, self).__init__()

        self.seq_len = seq_len
        self.long_term_multiplier = long_term_multiplier
        self.long_seq_len = seq_len * long_term_multiplier
        self.d_model = d_model
        self.input_c = input_c
        self.output_c = output_c
        self.device = device
        self.downsample_method = downsample_method
        self.upsample_method = upsample_method

        # Alpha configuration
        self.alpha_mode = alpha_mode
        if alpha_mode == 'learnable':
            # Initialize as learnable parameter (stored as logit)
            initial_logit = torch.log(torch.tensor(alpha_value) / (1 - torch.tensor(alpha_value)))
            self.alpha = nn.Parameter(initial_logit)
            print(f"Alpha mode: learnable (initialized to {torch.sigmoid(self.alpha).item():.4f})")
        else:
            # Register as fixed buffer (not learned)
            self.register_buffer('alpha', torch.tensor(alpha_value))
            print(f"Alpha mode: fixed (value = {alpha_value:.4f})")

        # Short-term model
        self.short_term_model = ShortTermModel(
            seq_len=seq_len,
            long_term_multiplier=long_term_multiplier,
            d_model=d_model,
            input_c=input_c,
            output_c=output_c,
            n_features=n_features,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            factor=factor,
            dropout=dropout,
            activation=activation,
            output_attention=output_attention,
            memory_initial=memory_initial,
            n_memory=n_memory,
            device=device,
            phase_type=f"{phase_type}_short",
            dataset_name=dataset_name,
            ucr_filename=ucr_filename,
            shrink_thres=shrink_thres,
            memory_init_embedding=memory_init_embedding,
            run_name=run_name,
            run_id=run_id
        )

        # Long-term model
        self.long_term_model = LongTermModel(
            seq_len=seq_len,
            long_term_multiplier=long_term_multiplier,
            d_model=d_model,
            input_c=input_c,
            output_c=output_c,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            factor=factor,
            dropout=dropout,
            activation=activation,
            output_attention=output_attention,
            memory_initial=memory_initial,
            n_memory=n_memory_long,
            device=device,
            phase_type=f"{phase_type}_long",
            dataset_name=dataset_name,
            ucr_filename=ucr_filename,
            shrink_thres=shrink_thres,
            memory_init_embedding=long_term_memory_init_embedding,
            run_name=run_name,
            run_id=run_id,
            downsample_method=downsample_method
        )

    def get_alpha(self):
        """Get alpha value (applies sigmoid for learnable mode)."""
        if self.alpha_mode == 'learnable':
            return torch.sigmoid(self.alpha)
        else:
            return self.alpha

    def forward(self, x_short=None, x_long=None, model_type='both', feature_short_reshaped=None):

        if model_type == 'short':
            return self.short_term_model(x_short, feature_short_reshaped)
        elif model_type == 'long':
            return self.long_term_model(x_long)
        elif model_type == 'both':
            short_out = self.short_term_model(x_short, feature_short_reshaped)
            long_out = self.long_term_model(x_long)
            return {
                'short': short_out,
                'long': long_out,
                'alpha': self.get_alpha()
            }
        else:
            raise ValueError(f"Unknown model_type: {model_type}")