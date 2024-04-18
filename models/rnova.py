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
        # ;pdb.set_trace()
        # print('pos_index', pos_index)
        mono_type = tgt[:, 1::2]  # Takes elements at even positions
        parent_index = tgt[:, 2::2]
        # print('mono_type', mono_type, 'parent_index', parent_index)
        mono_embeddings = self.tgt_token_embedding(mono_type)
        tgt_embeddings = torch.zeros((tgt.shape[0], tgt.shape[1], self.hidden_size)).cuda()
        # if odd_indices.shape[-1] > 0:
        tgt_embeddings[:, 0, :] = self.bos_embedding(tgt[:, 0])
        # idx_embeddings = self.idx_token_embedding(parent_index)
        # idx_position = torch.zeros(idx_embeddings.shape)
        # idx_position[:, :idx_embeddings.shape[1], :] = self.pos_embedding(parent_index)
        # print(idx_position.shape, mono_embeddings.shape)
        # TODO: at parent idx prediction use the embedding of that mono + predicted parent index
        tgt_embeddings[:, 1::2, :] = mono_embeddings.cuda()
        parent_index_expanded = parent_index.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        parent_mono_embedding = torch.gather(tgt_embeddings.clone(), 1, parent_index_expanded)
        tgt_embeddings[:, 2::2, :] = self.pos_embedding(parent_index) + parent_mono_embedding
        # if mono_type.shape[1] < parent_index.shape[1]:
        #     tgt_embeddings[:, 1::2, :] = mono_embeddings.cuda()
        #     tgt_embeddings[:, 1:-1:2, :] = tgt_embeddings[:, 1:-1:2, :]+ idx_position[:, :-1, :].cuda()
        # else:
        #     tgt_embeddings[:, 1::2, :] = mono_embeddings.cuda() + idx_position.cuda()
        # tgt = self.tgt_token_embedding(tgt)
        # print('tgt_embeddings', tgt_embeddings)
        tgt_embeddings = tgt_embeddings.cuda() + self.pos_embedding(pos_index)
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
        self.parent_idx_mass_embedding = SinusoidalPositionEmbedding(cfg.model.hidden_size // cfg.model.n_head)
        self.parent_batch_step_embedding = SinusoidalPositionEmbedding(cfg.model.hidden_size // cfg.model.n_head)
        self.mono_cross_mass_embedding = SinusoidalPositionEmbedding(cfg.model.key_size_decoder)
        self.parent_idx_cross_mass_embedding = SinusoidalPositionEmbedding(cfg.model.key_size_decoder)
        self.mono_linear_cross_mass = nn.Linear(cfg.model.key_size_decoder * 2, cfg.model.key_size_decoder)
        self.query_linear_mass = nn.Linear((cfg.model.hidden_size // cfg.model.n_head) * 2, cfg.model.hidden_size // cfg.model.n_head)

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
        self.parent_idx_mass_pool = nn.AdaptiveAvgPool1d(len(self.mass_list))

    def forward(self, logZ, src, tgt, pos_index, node_mass, rel_mask, glycan_mass, glycan_crossattn_mass):
        tgt_rep, _ = self.tgt_get(src=src, tgt=tgt, pos_index=pos_index, node_mass=node_mass,
                              rel_mask=rel_mask, glycan_mass=glycan_mass,
                              glycan_crossattn_mass=glycan_crossattn_mass)
        batch_size = tgt.shape[0]
        # valid_mask = torch.isfinite(tgt_rep)
        # sum =
        tgt_rep = F.softmax(tgt_rep, dim=-1)
        fwd_chain = logZ.repeat(batch_size)
        # print('forward', tgt_rep.shape, torch.roll(tgt, shifts=-1, dims=-1))
        s_p = torch.roll(tgt, shifts=-1, dims=-1).unsqueeze(-1)
        traj_logp = torch.gather(tgt_rep, 2, s_p).squeeze(-1)
        # print('result', traj_logp.shape)
        # traj_logp = tgt_rep[torch.roll(tgt, shifts=-1, dims=-1)]
        # print('traj', tgt, traj_logp, traj_logp.shape)
        output = torch.sum(traj_logp[:, :-1], dim=-1) + fwd_chain
        # print('output', output)
        return output

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

    def obtain_glycan_query_mass(self, glycan_mass):
        candidate_mass = glycan_mass.repeat_interleave(len(self.mass_list), dim=-1)
        candidates = torch.cat((self.mass_list, -self.mass_list)).expand_as(candidate_mass)
        candidate_mass += candidates
        candidate_mass = self.mono_mass_embedding(candidate_mass)
        return candidate_mass

    def obtain_glycan_query_cross_mass(self, glycan_crossattn_mass):
        query_glycan_cross_mass = glycan_crossattn_mass.repeat_interleave(len(self.mass_list), dim=-1)
        candidates = torch.cat((self.mass_list, self.mass_list)).expand_as(query_glycan_cross_mass)
        query_glycan_cross_mass += candidates
        query_glycan_cross_mass = self.mono_cross_mass_embedding(query_glycan_cross_mass)

        return query_glycan_cross_mass
    # @torch.autocast(device_type='cuda',dtype=torch.bfloat16)
    def tgt_get(self, *, src, tgt, pos_index, node_mass, rel_mask, glycan_mass, glycan_crossattn_mass, parent_mono_lists, label_mask_pad=None):
        # print('tgt_get pos_index', tgt, glycan_mass, glycan_crossattn_mass, parent_mono_lists)
        tgt_emb = self.tgt_seq_embedding(tgt, pos_index)
        tgt_in = self.tgt_in_out(tgt_emb)
        # tgt = self.tgt_embedding(tgt, pos_index)
        glycan_mass_emb = self.glycan_mass_embedding(glycan_mass).repeat(1, 1, self.cfg.model.n_head // 2, 1)
        glycan_query_mass = torch.zeros_like(glycan_mass_emb)
        glycan_crossattn_mass_emb = self.glycan_mass_embedding_cross(glycan_crossattn_mass).repeat(1, 1,
                                                                                                   self.cfg.model.n_head_per_mass_decoder,
                                                                                                   1)
        # print('glycan_mass_emb', glycan_mass_emb)
        seq_len = tgt.size(1)
        batch_size = tgt.size(0)
        query_glycan_cross_mass = torch.zeros_like(glycan_crossattn_mass_emb)
        mask_tensor = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        mask_tensor = mask_tensor.cuda()

        for i in range(seq_len):
            if i % 2 != 0:
                mask_tensor[i, :i] = torch.arange(i) % 2 != 0

        # if seq_len < 4: # only the first mono is connected to the special token
        mask_tensor[:, 0] = True
        # print('parent_mono_lists', parent_mono_lists.shape)
        for i in range(1, parent_mono_lists.shape[-1], 2):
            # print('parent_mono_lists',glycan_crossattn_mass, parent_mono_lists)
            parent_idx_step = glycan_crossattn_mass[:, 0:i:2, 1] + parent_mono_lists[:, i].unsqueeze(-1)
            # parent_mono_lists[:, i] += glycan_crossattn_mass[:, 2:i + 1:2]
            parent_mono_step = parent_mono_lists[:, 0].unsqueeze(-1)
            # if i != 2:
            parent_mono_step = torch.cat((parent_mono_step, parent_mono_lists[:, 1:i:2]),
                                         dim=1) + parent_mono_lists[:, i].unsqueeze(-1)
            # print('current tgt', tgt[:, :i+1], parent_mono_lists[:, i].unsqueeze(-1))
            # print('parent_mono_step', parent_mono_step)
            # print('parent_idx_step', parent_idx_step)
            parent_branch_step = torch.zeros_like(parent_mono_step)
            tgt_cur = tgt[:, :i+1]
            for p_i, p in enumerate(range(0,i+1,2)):
                masked_idx = ~mask_tensor[i, :i+1]
                masked_idx[-1] = False
                all_children_idx = torch.logical_and(tgt_cur == pos_index[:, p], masked_idx)
                # print('label_mask_pad', label_mask_pad, all_children_idx.shape)
                if label_mask_pad is not None:
                    all_children_idx = torch.logical_and(all_children_idx, label_mask_pad[:, :i+1])
                for b in range(batch_size):
                    all_children = tgt_cur[b][torch.roll(all_children_idx, -1)[b]]
                    # print('all_children', all_children)
                    if len(all_children) > 0:
                        parent_branch_step[b,p_i] = sum(self.id2mass[i.item()] for i in all_children)
            # print('parent_branch_step', parent_branch_step)
            parent_branch_step += parent_mono_lists[:, i].unsqueeze(-1)
            parent_branch_step_emb = self.parent_batch_step_embedding(parent_branch_step).transpose(1, 2)
            parent_branch_step_emb = self.parent_idx_mass_pool(parent_branch_step_emb).transpose(2, 1)
            # print('parent_batch_step', parent_branch_step)
            candidate_mass = self.parent_idx_mass_embedding(parent_mono_step).transpose(1, 2)
            candidate_mass = self.parent_idx_mass_pool(candidate_mass).transpose(2, 1)
            glycan_query_mass[:, i, :, :] = self.query_linear_mass(torch.concat((candidate_mass,parent_branch_step_emb),dim=-1).repeat(1, 1,
                                                                                              self.cfg.model.n_head // len(
                                                                                                  self.mass_list), 1))
            query_cross_mass = self.parent_idx_cross_mass_embedding(parent_idx_step).transpose(1, 2)
            query_cross_mass = self.parent_idx_mass_pool(query_cross_mass).transpose(2, 1).repeat(1, 1,
                                                                                                  self.cfg.model.n_head // len(
                                                                                                      self.mass_list),
                                                                                                  1)
            query_glycan_cross_mass[:, i, :, :] = query_cross_mass

        glycan_query_mass[:, 0::2, :, :] = self.obtain_glycan_query_mass(glycan_mass).repeat(1, 1,
                                                                                             self.cfg.model.n_head // (
                                                                                                     len(self.mass_list) * 2),
                                                                                             1)[:, 0::2, :, :]
        query_glycan_cross_mass[:, 0::2, :, :] = self.obtain_glycan_query_cross_mass(glycan_crossattn_mass).repeat(1, 1,
                                                                                                                   self.cfg.model.n_head // (
                                                                                                                           len(self.mass_list) * 2),
                                                                                                                   1)[:,0::2, :, :]
        glycan_crossattn_mass_emb = self.mono_linear_cross_mass(
            torch.cat((query_glycan_cross_mass, glycan_crossattn_mass_emb), dim=-1))
        node_mass = self.node_mass_decoder_embedding(node_mass).unsqueeze(2)
        batch_size = tgt.shape[0]
        tgt_mask = self.tgt_mask[:seq_len, :seq_len]
        rel_mask = self.combine_masks(seq_len, rel_mask)

        for l_decoder in self.decoder:
            tgt_emb = l_decoder(tgt_emb, mem=src,
                            glycan_mass=glycan_mass_emb,
                            glycan_query_mass=glycan_query_mass,
                            tgt_mask=tgt_mask,
                            node_mass=node_mass,
                            rel_mask=rel_mask.squeeze(-1).unsqueeze(1),
                            glycan_crossattn_mass=glycan_crossattn_mass_emb)

        is_idx = (torch.arange(pos_index.shape[1]) % 2 != 0).cuda()
        # print('mask_tensor', mask_tensor)
        tgt_sim = self.similarity_NN(tgt_emb)
        tgt_sim = torch.bmm(tgt_sim, tgt_in.transpose(1, 2))
        # print('tgt_sim', tgt_sim)
        tgt_sim = tgt_sim.masked_fill(~mask_tensor, float('-inf'))
        tgt_out = torch.ones((batch_size, seq_len, max(seq_len, self.output_size))).cuda() * float(
            '-inf')  # dtype=torch.bfloat16
        label_mask = torch.zeros((batch_size, seq_len, max(seq_len, self.output_size)),
                                 dtype=torch.bool).cuda()  # dtype=torch.bfloat16
        tgt_out[:, is_idx, :seq_len] = tgt_sim[:, is_idx, :]
        label_mask[:, is_idx, :seq_len] = mask_tensor[is_idx, :].unsqueeze(0).expand(batch_size, -1, -1)

        mono_out = self.output(tgt_emb)
        tgt_out[:, torch.logical_not(is_idx), :self.output_size] = mono_out[:, torch.logical_not(is_idx), :]
        label_mask[:, torch.logical_not(is_idx), :self.output_size] = 1
        if label_mask_pad is not None:
            label_mask[~label_mask_pad] = 0
        # print(label_mask)
        # tgt_out = self.custom_normalization(tgt_out)
        # print('tgt_out', tgt_out)
        return tgt_out, label_mask

    @torch.inference_mode()
    # @torch.autocast(device_type='cuda',dtype=torch.bfloat16)
    def inference_initialize(self, *, tgt, pos_index, node_mass, rel_mask, glycan_mass, mem, glycan_crossattn_mass):
        past_tgts = []
        past_mems = []
        tgt = self.tgt_token_embedding(tgt, pos_index)
        # print('tgt', tgt.shape)
        # print('inference_step tgt_seq', tgt.shape)

        glycan_mass = self.glycan_mass_embedding(glycan_mass)  # .unsqueeze(2)
        # print('glycan_mass', glycan_mass.shape)
        glycan_mass = glycan_mass.repeat(1, self.cfg.model.n_head // 2, 1).unsqueeze(1)
        # print('glycan_mass', glycan_mass.shape)
        glycan_crossattn_mass = self.glycan_mass_embedding_cross(glycan_crossattn_mass).repeat(1,
                                                                                               self.cfg.model.n_head_per_mass_decoder,
                                                                                               1).unsqueeze(1)
        node_mass = self.node_mass_decoder_embedding(node_mass).unsqueeze(2)
        for l_decoder in self.decoder:
            tgt, past_tgt, past_mem = l_decoder.inference_initialize(tgt, mem=mem,
                                                                     glycan_mass=glycan_mass,
                                                                     rel_mask=rel_mask.squeeze(-1).unsqueeze(1),
                                                                     # .unsqueeze(1),
                                                                     node_mass=node_mass,
                                                                     glycan_crossattn_mass=glycan_crossattn_mass)
            # print('past_tgt', past_tgt.shape)
            past_tgts.append(past_tgt)
            past_mems.append(past_mem)
        past_tgts = torch.stack(past_tgts, dim=1)
        past_mems = torch.stack(past_mems, dim=1)
        # print('past_tgts', past_tgts.shape, 'tgt', tgt.shape)
        # initialize will predict class type
        tgt_out = self.output(tgt)
        # print('tgt_out', tgt_out)
        return tgt_out, past_tgts, past_mems

    @torch.inference_mode()
    # @torch.autocast(device_type='cuda',dtype=torch.bfloat16)
    def inference_initialize_idx(self, *, tgt, pos_index, node_mass, rel_mask, glycan_mass, mem, glycan_crossattn_mass, tgt_mask):
        past_tgts = []
        past_mems = []
        true_idx = math.ceil(pos_index / 2)
        range_tensor = torch.arange(true_idx + 1)
        pos_index_array = range_tensor.repeat_interleave(2)[1:pos_index + 1].cuda().unsqueeze(0)

        tgt = self.tgt_seq_embedding(tgt, pos_index_array)
        tgt_in = self.tgt_in_out(tgt)
        # print('tgt in step', tgt.shape)
        # print('inference_step tgt_seq', tgt.shape)
        glycan_mass = self.glycan_mass_embedding(glycan_mass)  # .unsqueeze(2)
        # print('glycan_mass', glycan_mass.shape)
        glycan_mass = glycan_mass.repeat(1, 1, self.cfg.model.n_head // 2, 1)
        glycan_crossattn_mass = self.glycan_mass_embedding_cross(glycan_crossattn_mass).repeat(1, 1,
                                                                                               self.cfg.model.n_head_per_mass_decoder,
                                                                                               1)

        node_mass = self.node_mass_decoder_embedding(node_mass).unsqueeze(2)
        for l_decoder in self.decoder:
            tgt, past_tgt, past_mem = l_decoder.inference_initialize(tgt, mem=mem,
                                                                     glycan_mass=glycan_mass,
                                                                     rel_mask=rel_mask.squeeze(-1).unsqueeze(1),
                                                                     # .unsqueeze(1),
                                                                     node_mass=node_mass,
                                                                     glycan_crossattn_mass=glycan_crossattn_mass)
            # print('past_tgt', past_tgt.shape)
            past_tgts.append(past_tgt)
            past_mems.append(past_mem)
        past_tgts = torch.stack(past_tgts, dim=1)
        past_mems = torch.stack(past_mems, dim=1)
        # print('past_tgts', past_tgts.shape, 'tgt', tgt.shape)
        # initialize will predict class type
        tgt_sim = self.similarity_NN(tgt)
        # print('tgt_sim', tgt_sim.shape)
        tgt_sim = torch.bmm(tgt_sim, tgt_in.transpose(1, 2))
        # tgt_sim = tgt_sim[:, -1: -2,:]* tgt_sim.transpose(1, 2)
        # print('before step', tgt_sim.shape)
        # print('self.idx_mask[:pos_index + 1]', self.idx_mask[:pos_index + 1])
        cur_pos_index = copy.deepcopy(self.idx_mask[:pos_index])
        cur_pos_index[-1] = False # the last item of nonzero tgt
        tgt_out = tgt_sim.masked_fill(~cur_pos_index.unsqueeze(0), float('-inf'))
        # print('masked inference_initialize_idx', tgt_mask.shape, tgt_out.shape)
        tgt_out = tgt_out.masked_fill(~tgt_mask.unsqueeze(-1), float('-inf'))
        tgt_mask_expanded = tgt_mask.unsqueeze(1).expand_as(tgt_out)
        tgt_out = tgt_out.masked_fill(~tgt_mask_expanded, float('-inf'))

        # print('masked inference_initialize_idx', tgt_out, tgt_mask)
        # tgt_out = self.output(tgt)
        # print('tgt_out', tgt_out)
        return tgt_out, past_tgts, past_mems

    @torch.inference_mode()
    # @torch.autocast(device_type='cuda',dtype=torch.bfloat16)
    def inference_step(self, *, tgt, pos_index, past_tgts, past_mems, glycan_mass, rel_mask, glycan_crossattn_mass,
                       is_idx=None, tgt_mask=None):
        past_tgts_new = []
        if is_idx is None:
            is_idx = pos_index % 2 != 0
        # print('is_idx', is_idx)
        # print('inference_step tgt', tgt)
        true_idx = math.ceil(pos_index / 2)
        range_tensor = torch.arange(true_idx + 1)
        pos_index_array = range_tensor.repeat_interleave(2)[1:pos_index + 2].cuda().unsqueeze(0)

        tgt = self.tgt_seq_embedding(tgt, pos_index_array)
        tgt_in = self.tgt_in_out(tgt)
        # print('tgt in step', tgt.shape)
        # print('inference_step tgt_seq', tgt.shape)
        glycan_mass = self.glycan_mass_embedding(glycan_mass)  # .unsqueeze(2)
        # print('glycan_mass', glycan_mass.shape)
        glycan_mass = glycan_mass.repeat(1, 1, self.cfg.model.n_head // 2, 1)
        glycan_crossattn_mass = self.glycan_mass_embedding_cross(glycan_crossattn_mass).repeat(1, 1,
                                                                                               self.cfg.model.n_head_per_mass_decoder,
                                                                                               1)

        for i, l_decoder in enumerate(self.decoder):
            tgt, past_tgt_new = l_decoder.inference_step(tgt,
                                                         past_tgt=past_tgts[:, i],
                                                         past_mem=past_mems[:, i],
                                                         glycan_mass=glycan_mass,
                                                         rel_mask=rel_mask.squeeze(-1).unsqueeze(1),  # .unsqueeze(1),
                                                         glycan_crossattn_mass=glycan_crossattn_mass)
            past_tgts_new.append(past_tgt_new)
        past_tgts_new = torch.stack(past_tgts_new, dim=1)
        # print('step past_tgts', past_tgts.shape, 'tgt', tgt.shape)

        if is_idx:
            tgt_sim = self.similarity_NN(tgt)
            # print('tgt_sim', tgt_sim.shape)
            tgt_sim = torch.bmm(tgt_sim, tgt_in.transpose(1, 2))
            # tgt_sim = tgt_sim[:, -1: -2,:]* tgt_sim.transpose(1, 2)
            # print('before step', tgt_sim.shape)
            cur_pos_index = copy.deepcopy(self.idx_mask[:pos_index + 1])
            cur_pos_index[-1] = False
            tgt_out = tgt_sim.masked_fill(~cur_pos_index.unsqueeze(0), float('-inf'))
            if tgt_mask is not None:
                tgt_out = tgt_out.masked_fill(~tgt_mask.unsqueeze(-1), float('-inf'))
                tgt_mask_expanded = tgt_mask.unsqueeze(1).expand_as(tgt_out)
                tgt_out = tgt_out.masked_fill(~tgt_mask_expanded, float('-inf'))

        # print('masked inference step', tgt_out)
        else:
            tgt_out = self.output(tgt)
        # print('step tgt_out[0,]', tgt_out[0,])
        # print('out', tgt_out)
        return tgt_out, past_tgts_new


class RnovaBack(Rnova):
    def __init__(self, cfg, seq_max_len=64, aa_dict_size=8):
        super().__init__(cfg)
        self.tgt_seq_embedding = GlycanSeqIndexFirstEmbedding(cfg.model.hidden_size, seq_max_len, aa_dict_size)

    def inference_initialize(self, *, tgt, pos_index, node_mass, rel_mask, glycan_mass, mem, glycan_crossattn_mass):
        past_tgts = []
        past_mems = []
        # print('inference_step tgt_seq', tgt)

        tgt = self.tgt_seq_embedding(tgt, pos_index)
        glycan_mass = self.glycan_mass_embedding(glycan_mass)  # .unsqueeze(2)
        # print('glycan_mass', glycan_mass.shape)
        glycan_mass = glycan_mass.repeat(1, 1, self.cfg.model.n_head // 2, 1)
        # print('glycan_mass', glycan_mass.shape)
        glycan_crossattn_mass = self.glycan_mass_embedding_cross(glycan_crossattn_mass).repeat(1, 1,
                                                                                               self.cfg.model.n_head_per_mass_decoder,
                                                                                               1)
        node_mass = self.node_mass_decoder_embedding(node_mass).unsqueeze(2)
        for l_decoder in self.decoder:
            tgt, past_tgt, past_mem = l_decoder.inference_initialize(tgt, mem=mem,
                                                                     glycan_mass=glycan_mass,
                                                                     rel_mask=rel_mask.squeeze(-1).unsqueeze(1),
                                                                     # .unsqueeze(1),
                                                                     node_mass=node_mass,
                                                                     glycan_crossattn_mass=glycan_crossattn_mass)
            # print('past_tgt', past_tgt.shape)
            past_tgts.append(past_tgt)
            past_mems.append(past_mem)
        past_tgts = torch.stack(past_tgts, dim=1)
        past_mems = torch.stack(past_mems, dim=1)
        # print('past_tgts', past_tgts.shape, 'tgt', tgt.shape)
        # initialize will predict class type
        tgt_out = self.output(tgt)
        # print('tgt_out', tgt_out)
        return tgt_out, past_tgts, past_mems

    def tgt_get(self, *, src, tgt, pos_index, node_mass, rel_mask, glycan_mass, glycan_crossattn_mass):
        # print('tgt_get pos_index', tgt)
        tgt = self.tgt_seq_embedding(tgt, pos_index)
        tgt_in = self.tgt_in_out(tgt)
        # tgt = self.tgt_embedding(tgt, pos_index)
        glycan_mass = self.glycan_mass_embedding(glycan_mass).repeat(1, 1, self.cfg.model.n_head // 2, 1)
        glycan_crossattn_mass = self.glycan_mass_embedding_cross(glycan_crossattn_mass).repeat(1, 1,
                                                                                               self.cfg.model.n_head_per_mass_decoder,
                                                                                               1)
        # print('tgt', tgt.shape, glycan_mass.shape)
        node_mass = self.node_mass_decoder_embedding(node_mass).unsqueeze(2)
        seq_len = tgt.size(1)
        batch_size = tgt.shape[0]
        tgt_mask = self.tgt_mask[:seq_len, :seq_len]

        for l_decoder in self.decoder:
            tgt = l_decoder(tgt, mem=src,
                            glycan_mass=glycan_mass,
                            tgt_mask=tgt_mask,
                            node_mass=node_mass,
                            rel_mask=rel_mask.squeeze(-1).unsqueeze(1),
                            glycan_crossattn_mass=glycan_crossattn_mass)
        is_idx = (torch.arange(pos_index.shape[1]) % 2 == 0).cuda()
        mask_tensor = torch.zeros(seq_len, seq_len, dtype=torch.bool)

        for i in range(seq_len):
            if i % 2 == 0:
                mask_tensor[i, :i + 1] = torch.arange(i + 1) % 2 != 0
        # if seq_len < 4: # only the first mono is connected to the special token
        mask_tensor[:, 0] = True
        mask_tensor[:, -1] = False
        # print('mask_tensor', mask_tensor)
        mask_tensor = mask_tensor.cuda()
        tgt_sim = self.similarity_NN(tgt)
        tgt_sim = torch.bmm(tgt_sim, tgt_in.transpose(1, 2))
        tgt_sim = tgt_sim.masked_fill(~mask_tensor, float('-inf'))
        tgt_out = torch.ones((batch_size, seq_len, max(seq_len, self.output_size))).cuda() * float(
            '-inf')  # dtype=torch.bfloat16
        label_mask = torch.zeros((batch_size, seq_len, max(seq_len, self.output_size)),
                                 dtype=torch.bool).cuda()  # dtype=torch.bfloat16
        tgt_out[:, is_idx, :seq_len] = tgt_sim[:, is_idx, :]
        label_mask[:, is_idx, :seq_len] = mask_tensor[is_idx, :].unsqueeze(0).expand(batch_size, -1, -1)

        mono_out = self.output(tgt)
        tgt_out[:, torch.logical_not(is_idx), :self.output_size] = mono_out[:, torch.logical_not(is_idx), :]
        label_mask[:, torch.logical_not(is_idx), :self.output_size] = 1
        # print(label_mask)
        # print('tgt_out', list(tgt_out), )
        # tgt_out = self.custom_normalization(tgt_out)
        return tgt_out, label_mask