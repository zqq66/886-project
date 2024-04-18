import copy

import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.cuda.amp import autocast
from gnova.modules.decoderlayer import DecoderLayer
from gnova.modules import GNovaEncoderLayer


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim, lambda_max=1e4, lambda_min=1e-5) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        base = lambda_max / (2 * np.pi)
        scale = lambda_min / lambda_max
        div_term = torch.from_numpy(base * scale ** (np.arange(0, dim, 2) / dim))
        self.register_buffer('div_term', div_term)

    def forward(self, mass_position):
        pe_sin = torch.sin(mass_position.unsqueeze(dim=-1) / self.div_term)
        pe_cos = torch.cos(mass_position.unsqueeze(dim=-1) / self.div_term)
        return torch.concat([pe_sin, pe_cos], dim=-1).float()


class GlycanSeqIndexFirstEmbedding(nn.Module):
    def __init__(self, hidden_size, max_len, aa_dict_size):
        super().__init__()
        self.tgt_token_embedding = nn.Embedding(aa_dict_size, hidden_size)
        self.idx_token_embedding = nn.Embedding(max_len, hidden_size)
        self.pos_embedding = SinusoidalPositionEmbedding(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, tgt, pos_index):

        # print('mono_type', tgt[:, 2::2])
        mono_type = tgt[:, 2::2]  # Takes elements at even positions
        parent_index = tgt[:, 1::2]
        # print('mono_type', mono_type, 'parent_index', parent_index)
        mono_embeddings = self.tgt_token_embedding(mono_type).cuda()
        tgt_embeddings = torch.zeros((tgt.shape[0], tgt.shape[1], self.hidden_size)).cuda()
        # if odd_indices.shape[-1] > 0:
        idx_embeddings = self.idx_token_embedding(parent_index).cuda()
        idx_position = torch.zeros(idx_embeddings.shape).cuda()
        idx_position[:, :idx_embeddings.shape[1], :] = self.pos_embedding(parent_index)
        tgt_embeddings[:, 1::2, :] = idx_embeddings
        if mono_type.shape[1] < parent_index.shape[1]:
            tgt_embeddings[:, 2::2, :] = mono_embeddings + idx_position[:, :-1, :]

        else:
            tgt_embeddings[:, 2::2, :] = mono_embeddings + idx_position
        # tgt = self.tgt_token_embedding(tgt)
        # print('tgt_embeddings', tgt_embeddings.shape)
        tgt_embeddings[:, 0, :] = self.tgt_token_embedding(tgt[:, 0]).cuda()
        tgt_embeddings = tgt_embeddings + self.pos_embedding(pos_index)
        return tgt_embeddings


class GlycanSeqEmbedding(nn.Module):
    def __init__(self, hidden_size, max_len, aa_dict_size):
        super().__init__()
        self.bos_embedding = nn.Embedding(1, hidden_size)
        self.tgt_token_embedding = nn.Embedding(aa_dict_size, hidden_size)
        self.pos_embedding = SinusoidalPositionEmbedding(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, tgt, pos_index):
        mono_embeddings = self.tgt_token_embedding(tgt)
        tgt_embeddings = mono_embeddings.cuda() + self.pos_embedding(pos_index)
        return tgt_embeddings


class GlycanTokenEmbedding(nn.Module):
    def __init__(self, hidden_size, max_len, aa_dict_size):
        super().__init__()
        self.tgt_token_embedding = nn.Embedding(aa_dict_size, hidden_size)
        self.idx_token_embedding = nn.Embedding(max_len, hidden_size)
        self.pos_embedding = SinusoidalPositionEmbedding(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, tgt, pos_index):

        if pos_index % 2 == 0:
            # print(tgt, pos_index)
            tgt = self.tgt_token_embedding(tgt)
        else:
            tgt = self.idx_token_embedding(tgt)
        pos_index = torch.ceil(pos_index / 2).to(torch.long)
        tgt = tgt.cuda() + self.pos_embedding(pos_index)
        return tgt


class SpectraEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.peak_embedding = nn.Linear(hidden_size, hidden_size)
        self.charge_embedding = nn.Embedding(10, hidden_size)
        self.additional_peak_embedding = nn.Embedding(3, hidden_size)

    def forward(self, src, charge, index):
        src = self.peak_embedding(src) + self.charge_embedding(charge).unsqueeze(1) + self.additional_peak_embedding(
            index)
        return src


class Rnova(nn.Module):
    # output size: 5 mono residues + bos
    def __init__(self, cfg, mass_list, id2mass, seq_max_len=64, aa_dict_size=8, output_size=5) -> None:
        super().__init__()
        self.cfg = cfg
        alpha_decoder = 1.316 * cfg.model.num_layers ** (1 / 4)
        beta_decoder = 0.5373 * cfg.model.num_layers ** (-1 / 4)
        self.id2mass = id2mass
        tgt_mask = torch.ones((seq_max_len, seq_max_len), dtype=bool).tril()
        self.register_buffer('tgt_mask', tgt_mask, persistent=False)
        self.idx_mask = (torch.arange(seq_max_len) % 2 != 0).cuda()
        self.idx_mask[0] = True

        self.node_feature_proj = nn.Linear(9, cfg.model.hidden_size)
        self.node_sourceion_embedding = nn.Embedding(20, cfg.model.hidden_size, padding_idx=0)
        self.node_mass_embedding = SinusoidalPositionEmbedding(cfg.model.d_relation)
        self.node_mass_decoder_embedding = SinusoidalPositionEmbedding(cfg.model.key_size_decoder)
        self.glycan_mass_embedding_cross = SinusoidalPositionEmbedding(cfg.model.key_size_decoder)
        self.glycan_mass_embedding = SinusoidalPositionEmbedding(cfg.model.hidden_size // cfg.model.n_head)
        self.mono_mass_embedding = SinusoidalPositionEmbedding(cfg.model.hidden_size // cfg.model.n_head)
        self.glycan_comp_mass_embedding = SinusoidalPositionEmbedding(cfg.model.hidden_size // cfg.model.n_head)
        self.glycan_comp_mass_embedding_cross = SinusoidalPositionEmbedding(cfg.model.key_size_decoder)

        self.query_linear_mass = nn.Linear(cfg.model.key_size_decoder * 2,cfg.model.key_size_decoder)

        self.encoder = nn.ModuleList([GNovaEncoderLayer(hidden_size=cfg.model.hidden_size,
                                                        d_relation=cfg.model.d_relation,
                                                        alpha=(2 * cfg.model.num_layers) ** 0.25,
                                                        beta=(8 * cfg.model.num_layers) ** -0.25,
                                                        dropout_rate=cfg.model.dropout_rate) \
                                      for _ in range(cfg.model.num_layers)])

        self.tgt_token_embedding = GlycanTokenEmbedding(cfg.model.hidden_size, seq_max_len, aa_dict_size)
        self.tgt_seq_embedding = GlycanSeqEmbedding(cfg.model.hidden_size, seq_max_len, aa_dict_size)

        self.decoder = nn.ModuleList([DecoderLayer(cfg.model.hidden_size,
                                                   cfg.model.n_head,
                                                   cfg.model.max_charge,
                                                   cfg.model.n_head_per_mass_decoder,
                                                   cfg.model.key_size_decoder,
                                                   alpha_decoder,
                                                   beta_decoder,
                                                   cfg.model.decoder_dropout_rate) \
                                      for _ in range(cfg.model.num_layers)])

        self.output = nn.Sequential(nn.Linear(cfg.model.hidden_size, cfg.model.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(cfg.model.hidden_size, output_size))

        self.similarity_NN = nn.Linear(cfg.model.hidden_size, cfg.model.hidden_size)
        self.similarity_out = nn.Linear(cfg.model.hidden_size, 1)
        self.tgt_in_out = nn.Linear(cfg.model.hidden_size, cfg.model.hidden_size)
        # self.layernorm = nn.LayerNorm(seq_max_len, elementwise_affine=False)
        self.output_size = output_size
        self.mass_list = mass_list
        self.parent_idx_mass_pool = nn.AdaptiveAvgPool1d(self.cfg.model.n_head)


    # @torch.autocast(device_type='cuda',dtype=torch.bfloat16)
    def mem_get(self, node_feature, node_sourceion, node_mass, dist, predecessors, rel_mask):
        node = self.node_feature_proj(node_feature) + self.node_sourceion_embedding(node_sourceion)
        node_mass = self.node_mass_embedding(node_mass)
        for l_encoder in self.encoder:
            src = l_encoder(node, node_mass, dist, predecessors, rel_mask)
        return src

    def combine_masks(self, seq_len, memory_padding_mask):
        """
        Combines a causal mask with a memory padding mask.
        """
        # Expand memory_padding_mask to match the dimensions of causal_mask

        batch_size = memory_padding_mask.shape[0]
        causal_mask = torch.triu(torch.ones(memory_padding_mask.shape[2], seq_len, dtype=torch.bool))
        # causal_mask = ~causal_mask
        causal_mask = causal_mask.repeat(batch_size, 1, 1).cuda()
        # print(causal_mask[:,:4, :4])

        expanded_memory_padding_mask = memory_padding_mask.squeeze(1).expand_as(causal_mask)

        # Combine the masks using logical OR operation
        combined_mask = causal_mask & expanded_memory_padding_mask

        return combined_mask.transpose(-2, -1)

    def obtain_parent_combine_mass(self, glycan_mass):
        combine_mass = [{0, glycan_mass[0]}]
        for i, m in enumerate(glycan_mass[1:]):
            i = i + 1
            current_mass = set(m_t + m for m_t in combine_mass[i - 1])
            current_mass.add(m)
            current_mass = current_mass.union(combine_mass[i - 1])
            combine_mass.append(current_mass)
        return combine_mass

    # @torch.autocast(device_type='cuda',dtype=torch.bfloat16)
    def tgt_get(self, *, src, tgt, pos_index, node_mass, rel_mask, glycan_mass, glycan_crossattn_mass, parent_mono_lists, label_mask_pad=None):

        tgt_emb = self.tgt_seq_embedding(tgt, pos_index)
        # tgt = self.tgt_embedding(tgt, pos_index)
        glycan_mass_emb = self.glycan_mass_embedding(glycan_mass).repeat(1, 1, self.cfg.model.n_head // 2, 1)
        glycan_crossattn_mass_emb = self.glycan_mass_embedding_cross(glycan_crossattn_mass).repeat(1, 1,
                                                                                                   self.cfg.model.n_head_per_mass_decoder,
                                                                                                   1)
        batch_size = tgt.shape[0]
        mass_comb_emb_pools_b = []
        mass_comb_emb_pools_cross_b = []

        for b in range(batch_size):
            glycan_mass_comb = self.obtain_parent_combine_mass(parent_mono_lists[b, :].tolist())
            mass_comb_emb_pools = []
            mass_comb_emb_pools_cross = []
            for i in glycan_mass_comb:
                mass_comb = torch.tensor(list(i)).cuda()
                mass_comb_emb = self.glycan_comp_mass_embedding(mass_comb).transpose(0,1)
                mass_comb_emb_cross = self.glycan_comp_mass_embedding_cross(mass_comb).transpose(0,1)
                mass_comb_emb_pool = self.parent_idx_mass_pool(mass_comb_emb).transpose(1,0)
                mass_comb_emb_pool_cross = self.parent_idx_mass_pool(mass_comb_emb_cross).transpose(1,0)
                mass_comb_emb_pools.append(mass_comb_emb_pool)
                mass_comb_emb_pools_cross.append(mass_comb_emb_pool_cross)
            mass_comb_emb_pools = torch.stack(mass_comb_emb_pools)
            mass_comb_emb_pools_cross = torch.stack(mass_comb_emb_pools_cross)
            mass_comb_emb_pools_b.append(mass_comb_emb_pools)
            mass_comb_emb_pools_cross_b.append(mass_comb_emb_pools_cross)

        mass_comb_emb_pools_b = torch.stack(mass_comb_emb_pools_b)
        mass_comb_emb_pools_cross_b = torch.stack(mass_comb_emb_pools_cross_b)
        glycan_crossattn_comp_mass_emb = torch.concat((glycan_crossattn_mass_emb, mass_comb_emb_pools_cross_b), dim=-1)
        glycan_crossattn_comp_mass_emb = self.query_linear_mass(glycan_crossattn_comp_mass_emb)
        # print('glycan_mass_emb', glycan_mass_emb)
        seq_len = tgt.size(1)

        node_mass = self.node_mass_decoder_embedding(node_mass).unsqueeze(2)
        tgt_mask = self.tgt_mask[:seq_len, :seq_len]
        rel_mask = self.combine_masks(seq_len, rel_mask)

        for l_decoder in self.decoder:
            tgt_emb = l_decoder(tgt_emb, mem=src,
                            glycan_mass=mass_comb_emb_pools_b,
                            glycan_query_mass=mass_comb_emb_pools_b,
                            tgt_mask=tgt_mask,
                            node_mass=node_mass,
                            rel_mask=rel_mask.squeeze(-1).unsqueeze(1),
                            glycan_crossattn_mass=glycan_crossattn_comp_mass_emb)
        mono_out = self.output(tgt_emb)

        # print(label_mask)
        # tgt_out = self.custom_normalization(tgt_out)
        # print('tgt_out', tgt_out)
        return mono_out