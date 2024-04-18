import copy

import torch
from dataclasses import dataclass, field
from gnova.utils.BasicClass import Residual_seq, Composition
import operator
from copy import deepcopy
import math
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist


class Inference_label(object):
    def __init__(self, cfg, model, inference_dl, aa_dict, tokenize_aa_dict, detokenize_aa_dict, knapsack_mask):
        self.cfg = cfg
        self.model = model
        self.inference_dl_ori = inference_dl
        self.knapsack_mask_mass = knapsack_mask['mass']
        self.knapsack_mask_aa = knapsack_mask['aa_composition']
        self.id2mass = {i:m for i, m in enumerate(self.knapsack_mask_mass)}

        self.aa_dict = aa_dict
        self.tokenize_aa_dict = tokenize_aa_dict
        self.detokenize_aa_dict = detokenize_aa_dict
        self.prediction_rate = []
        self.unable2predict = set()
        self.range_tensor = torch.arange(self.cfg.data.peptide_max_len)[1::2]
        self.range_tensor = self.range_tensor.repeat_interleave(2)
        self.range_tensor = torch.concat((torch.tensor([0]), self.range_tensor))

    def __iter__(self):
        self.inference_dl = iter(self.inference_dl_ori)
        return self

    def __next__(self):
        encoder_input, decoder_input, seq, precursor_mass, pep_mass, psm_index, _, label, label_mask = next(self.inference_dl)
        if dist.is_initialized():
            mem = self.model.module.mem_get(**encoder_input)
        else:
            mem = self.model.mem_get(**encoder_input)
        finishing_seq = []
        batch_size = len(seq)
        # initializing
        tgt = torch.zeros((batch_size, 1), dtype=torch.long).cuda()
        bos_tgt = copy.deepcopy((tgt))
        glycan_mass = torch.zeros((batch_size,1, 2)).cuda()
        glycan_mass[:,0, 1] = torch.tensor(precursor_mass)
        glycan_crossattn_mass = torch.zeros((batch_size,1, 2)).cuda()
        parent_mono_lists = torch.zeros((batch_size,1)).cuda()
        pos_index = 0
        node_mass = encoder_input['node_mass']
        rel_mask = encoder_input['rel_mask']
        min_mono = min(self.knapsack_mask_mass)
        available_index = torch.full((batch_size,), True).cuda()

        while len(finishing_seq) < batch_size:
            # true_idx = math.ceil((tgt.shape[-1]-1) / 2)
            # print('tgt.shape', tgt)

            # print('range_tensor',self.range_tensor[:tgt.shape[1]].cuda().unsqueeze(0))
            with torch.no_grad():
                next_dist, _ = self.model.tgt_get(src=mem,
                                               tgt=tgt,
                                               pos_index=self.range_tensor[:tgt.shape[1]].cuda().unsqueeze(0),
                                               node_mass=node_mass,
                                               rel_mask=rel_mask,
                                               glycan_mass=glycan_mass[available_index],
                                               glycan_crossattn_mass=glycan_crossattn_mass[available_index],
                                               parent_mono_lists=parent_mono_lists[available_index])
            next_tgt = torch.argmax(next_dist, dim=-1)
            # if next_tgt.shape[-1] == 5:
            #     next_tgt[:, -1] = 4
            # tgt = torch.concat((tgt, next_tgt[:,-1:]), dim=-1)
            tgt = torch.concat((bos_tgt, next_tgt), dim=-1)

            # print('tgt', tgt, next_tgt, next_dist[:, -1, :])
            glycan_mass,glycan_crossattn_mass, available_index, parent_mono_lists = self.obtain_glycan_mass(tgt, precursor_mass)
            mem = mem[available_index]
            rel_mask = rel_mask[available_index]
            bos_tgt= bos_tgt[available_index]

            pos_index += 1
            # print(torch.any(~available_index), available_index)
            if torch.any(~available_index):
                list_index = torch.where(~available_index)[0]
                print('idx', available_index, tgt.shape, list_index, seq, len(psm_index))

                for i in list_index:
                    finishing_seq.append((tgt[i], seq[i],psm_index[i]))
                    mass_block = node_mass[i][label[i] > 0]
                    mass_block = mass_block[mass_block >= min_mono].cpu()
                    print('mass_block', mass_block)
            node_mass = node_mass[available_index]
            label = label[available_index]
            seq = [s for i,s in enumerate(seq) if i in torch.where(available_index)[0].tolist()]
            psm_index = [p for i,p in enumerate(psm_index) if i in torch.where(available_index)[0].tolist()]
            tgt = tgt[available_index]

        self.prediction_rate.append(len(finishing_seq)/batch_size)
        return finishing_seq

    def obtain_parent_mass(self, mass_list, tgt):
        mass_list_parent = torch.cumsum(mass_list, dim=0)
        for idx, mass in enumerate(mass_list):
            if idx % 2 != 0 and idx > 0:  # mono
                mass_list_parent[idx] = mass
            elif idx > 2:  # parent idx
                mass_list_parent[idx] = mass_list_parent[tgt[idx] + 1] + \
                                        self.id2mass[tgt[idx - 1].item()]
        return mass_list_parent

    def obtain_glycan_mass(self, next_tgts, precursor_mass):
        glycan_mass = []
        glycan_crossattn = []
        parent_mono_lists = []
        batch_size = next_tgts.shape[0]
        available_index = torch.full((batch_size,), True)
        for i,tgt in enumerate(next_tgts):
            mass_list = torch.zeros((len(tgt)))
            mass_list[1::2] = torch.tensor(
                [self.detokenize_aa_dict[i.item()] for i in tgt[1::2]])
            nmass = torch.cumsum(mass_list, dim=0)
            # if len(tgt) == 19:
            #     available_index[i] = False
            if len(tgt) % 2 == 1:
                if abs((torch.sum(mass_list)) - precursor_mass[i]) < 0.04 or torch.sum(mass_list)>precursor_mass[i]:
                    print(abs(torch.sum(mass_list) - precursor_mass[i])<0.04)
                    available_index[i] = False
            pmass = torch.tensor(self.obtain_parent_mass(mass_list, tgt))  # current_status.precursor_mass - nmass
            cmass = precursor_mass[i] - nmass
            glycan_mass_item = torch.stack([nmass, cmass], dim=-1)
            glycan_crossattn_item = torch.stack([nmass, pmass], dim=-1)
            glycan_mass.append(glycan_mass_item)
            glycan_crossattn.append(glycan_crossattn_item)
            parent_mono_lists.append(mass_list)
        glycan_mass = torch.stack(glycan_mass)
        glycan_crossattn = torch.stack(glycan_crossattn)
        parent_mono_lists = torch.stack(parent_mono_lists)
        return glycan_mass.cuda(), glycan_crossattn.cuda(), available_index.cuda(), parent_mono_lists.cuda()




