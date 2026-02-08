from __future__ import absolute_import, print_function
import os
import torch
import torch.nn as nn
from torch.nn import functional as F


class MemoryModule(nn.Module):

    def __init__(self, n_memory, fea_dim, n_vars, shrink_thres=0.0025, device=None,
                 memory_init_embedding=None, phase_type=None, dataset_name=None,
                 file_name=None, run_name=None, run_id=None):
        super(MemoryModule, self).__init__()

        self.n_memory = n_memory
        self.fea_dim = fea_dim
        self.n_vars = n_vars
        self.shrink_thres = shrink_thres
        self.phase_type = phase_type
        self.device = device

        # Update gate layers (shared across all variables)
        self.U = nn.Linear(fea_dim, fea_dim)
        self.W = nn.Linear(fea_dim, fea_dim)

        # Initialize memory bank: (N, M, C)
        self._initialize_memory_bank(
            memory_init_embedding, phase_type, dataset_name,
            file_name, run_name, run_id
        )

    def _initialize_memory_bank(self, memory_init_embedding, phase_type,
                                dataset_name, file_name, run_name, run_id):
        """Initialize memory bank based on phase type."""
        if memory_init_embedding is None:
            if phase_type == 'test_short':
                # Test phase: load saved memory
                if dataset_name == 'UCR_Anomaly':
                    file_name_without_ext = os.path.splitext(file_name)[0]
                    load_path = f'save_models/VLoLo/{run_name}/{run_id}/memory_item/{dataset_name}/{file_name_without_ext}/{dataset_name}_memory_item.pth'
                else:
                    load_path = f'save_models/VLoLo/{run_name}/{run_id}/memory_item/{dataset_name}_memory_item.pth'

                loaded_mem = torch.load(load_path)

                if loaded_mem.dim() == 3:
                    init_mem = loaded_mem
                else:
                    init_mem = loaded_mem.unsqueeze(0).repeat(self.n_vars, 1, 1)

                print(f'{load_path} - loading memory banks for {self.n_vars} variables (test phase)')
                self.register_buffer('mem_bank', init_mem)
            else:
                # First training phase: random initialization
                print(f'Initializing memory banks randomly for {self.n_vars} variables (first train phase)')
                init_mem = F.normalize(
                    torch.rand((self.n_vars, self.n_memory, self.fea_dim), dtype=torch.float),
                    dim=-1
                )
                self.register_buffer('mem_bank', init_mem)
        else:
            # Second training phase: initialize with provided embedding
            if memory_init_embedding.dim() == 3:
                init_mem = memory_init_embedding
            else:
                init_mem = memory_init_embedding.unsqueeze(0).repeat(self.n_vars, 1, 1)
            self.register_buffer('mem_bank', init_mem)

    def hard_shrink_relu(self, input, lambd=0.0025, epsilon=1e-12):
        """Hard shrink with ReLU activation."""
        output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)
        return output

    def get_read_attn_batched(self, query):
        """
        Args:
            query: (B, N, C)

        Returns:
            attn: (B, N, M)
        """
        attn = torch.einsum('bnc,nmc->bnm', query, self.mem_bank.detach())
        attn = F.softmax(attn, dim=-1)

        if self.shrink_thres > 0:
            attn = self.hard_shrink_relu(attn, self.shrink_thres)
            attn = F.normalize(attn, p=1, dim=-1)

        return attn

    def read_batched(self, query):
        """
        Args:
            query: (B, N, C)

        Returns:
            dict: output (B, N, 2C), attn (B, N, M)
        """
        attn = self.get_read_attn_batched(query)
        add_memory = torch.einsum('bnm,nmc->bnc', attn, self.mem_bank.detach())
        read_query = torch.cat([query, add_memory], dim=-1)

        return {'output': read_query, 'attn': attn}

    def get_update_attn_batched(self, query):
        """
        Args:
            query: (B, N, C)

        Returns:
            attn: (N, M, B)
        """
        query_transposed = query.permute(1, 2, 0)  # (N, C, B)
        attn = torch.bmm(self.mem_bank, query_transposed)  # (N, M, B)
        attn = F.softmax(attn, dim=-1)

        if self.shrink_thres > 0:
            attn = self.hard_shrink_relu(attn, self.shrink_thres)
            attn = F.normalize(attn, p=1, dim=-1)

        return attn

    def update_batched(self, query):
        """
        Batched memory update operation.

        Args:
            query: (B, N, C)
        """
        attn = self.get_update_attn_batched(query.detach())
        query_permuted = query.detach().permute(1, 0, 2)  # (N, B, C)
        add_mem = torch.bmm(attn, query_permuted)  # (N, M, C)

        # Update gate
        update_gate = torch.sigmoid(self.U(self.mem_bank) + self.W(add_mem))

        # Update memory bank
        self.mem_bank.data = (1 - update_gate) * self.mem_bank + update_gate * add_mem

    def forward(self, query):
        """
        Args:
            query: (B, N, C)

        Returns:
            dict:
                - output: (B, N, 2C)
                - attn: (B, N, M)
                - memory_init_embedding: (N, M, C)
        """
        is_training = (self.phase_type != 'test_short')

        if is_training:
            self.update_batched(query)

        outs = self.read_batched(query)

        return {
            'output': outs['output'],
            'attn': outs['attn'],
            'memory_init_embedding': self.mem_bank
        }

    @property
    def mem(self):
        """Get memory bank for all variables."""
        return self.mem_bank