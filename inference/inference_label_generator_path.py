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
from gnova.data.label_generator import unique_int


class Inference_label_path(object):
    def __init__(self, cfg, model, inference_dl, aa_dict, tokenize_aa_dict, detokenize_aa_dict, knapsack_mask):
        self.cfg = cfg
        self.model = model
        self.inference_dl_ori = inference_dl
        self.knapsack_mask_mass = knapsack_mask['mass']
        self.knapsack_mask_aa = knapsack_mask['aa_composition']
        self.id2mass = {i:m for i, m in enumerate(self.knapsack_mask_mass)}
        self.mass2id = {int(m):i for i, m in self.id2mass.items()}
        self.knapsack_mask_mass = np.sort(self.knapsack_mask_mass)

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
        mass_tags, parent_mass = self.obtain_optimal_path(encoder_input, label)
        print('mass_tags, parent_mass', mass_tags, parent_mass)
        while len(finishing_seq) < batch_size:
            # true_idx = math.ceil((tgt.shape[-1]-1) / 2)
            # print('tgt.shape', tgt)

            # print('range_tensor',self.range_tensor[:tgt.shape[1]].cuda().unsqueeze(0))
            with torch.no_grad():
                next_dist, _ = self.model.tgt_get(src=mem,
                                               tgt=tgt[available_index],
                                               pos_index=self.range_tensor[:tgt.shape[1]].cuda().unsqueeze(0),
                                               node_mass=node_mass,
                                               rel_mask=rel_mask,
                                               glycan_mass=glycan_mass[available_index],
                                               glycan_crossattn_mass=glycan_crossattn_mass[available_index],
                                               parent_mono_lists=parent_mono_lists[available_index])
            next_tgt = torch.argmax(next_dist, dim=-1)
            # if next_tgt.shape[-1] == 5:
            #     next_tgt[:, -1] = 4
            # when it's mono type to be predicted
            # check parent mass, if the mono is different than the prediction
            # and the mono never existed in the predicted structure
            # replace the prediction with restricted mono
            old_tgt = copy.deepcopy(tgt)
            tgt = torch.concat((tgt, next_tgt[:, -1:]), dim=-1)
            if next_tgt.shape[-1] % 2 == 1:
                for b in range(next_tgt.shape[0]):
                    # only matched entire substructure
                    mono_constrain = self.obtain_mass_constrain(mass_tags[b], parent_mass[b], glycan_mass[b, -1, 0].cpu().numpy())
                    if mono_constrain == -1:
                        tgt[b] = torch.concat((old_tgt[b], next_tgt[b, -1:]), dim=-1)
                    else:
                        predicted_mono = next_tgt[b, -1]
                        # print('mono_constrain', mono_constrain)
                        if mono_constrain != predicted_mono:
                            print('mono_constrain applied', mono_constrain, predicted_mono, glycan_mass[b, -1, 0])
                            mono_constrain = torch.tensor([mono_constrain]).cuda()
                            tgt[b] = torch.concat((old_tgt[b], mono_constrain), dim=-1)
                            print('tgt', tgt, next_tgt, next_dist[:, -1, :])
                        else:
                            # print('tgt[b], next_tgt[b, -1]', tgt[b], next_tgt[b, -1:])
                            tgt[b] = torch.concat((old_tgt[b], next_tgt[b, -1:]), dim=-1)
            glycan_mass,glycan_crossattn_mass, available_index, parent_mono_lists = self.obtain_glycan_mass(tgt, precursor_mass)
            mem = mem[available_index]
            rel_mask = rel_mask[available_index]
            bos_tgt= bos_tgt[available_index]

            pos_index += 1
            # print(torch.any(~available_index), available_index)
            if torch.any(~available_index):
                # print('idx', available_index, tgt)
                list_index = torch.where(~available_index)[0]
                for i in list_index:
                    finishing_seq.append((tgt[i], seq[i],psm_index[i]))
                    mass_block = node_mass[i][label[i] > 0]
                    mass_block = mass_block[mass_block >= min_mono].cpu()
                    print('mass_block', mass_block)
            node_mass = node_mass[available_index]
            label = label[available_index]

        self.prediction_rate.append(len(finishing_seq)/batch_size)
        return finishing_seq

    def obtain_mass_constrain(self, mass_tags, parent_masses, path_mass):
        # even based on latest mass, the next prediction may not rely on this mass
        # knapsack should be obtained when idx is predicted
        # if the predicted mono doesn't meet the mass find second best prediction
        # print('no', parent_masses.shape)
        if not parent_masses.shape:
            return -1
        path_mass_match_left = parent_masses.searchsorted(path_mass - 0.04)
        path_mass_match_right = parent_masses.searchsorted(path_mass + 0.04)
        mass_tag = mass_tags[path_mass_match_left:path_mass_match_right].astype(int).tolist()
        if len(mass_tag) < 1:
            return -1
        mass_tag_idx = self.mass2id[mass_tag[0]]
        return mass_tag_idx

    def find_if_list_mass_is_mono(self, mass_list, reference):
        ms2_left_boundary = reference.searchsorted(mass_list - 0.04, side='left')
        ms2_right_boundary = reference.searchsorted(mass_list + 0.04, side='right')
        return torch.tensor(ms2_right_boundary>ms2_left_boundary)

    def obtain_optimal_path(self, encoder_input, label):
        # TODO: retrain the classification model
        # pred = self.model_knapsack(encoder_input)
        # pred = torch.sigmoid(pred).squeeze(-1)
        mass_tags = []
        parent_mass = []
        mass_blocks = []
        min_mono = min(self.knapsack_mask_mass)

        for i in range(label.shape[0]):
            mass_block = (encoder_input['node_mass'][i])[label[i]>0.5]
            mass_block = mass_block[mass_block >= min_mono].cpu()
            mass_block = unique_int(mass_block).numpy()

            mass_block = np.append(0, mass_block)
            parent_idx = torch.arange(mass_block.shape[0])
            print('mass_block', mass_block)
            mass_tag = np.zeros_like(mass_block)
            diff_mass = np.diff(mass_block)
            mass_exist = self.find_if_list_mass_is_mono(diff_mass, self.knapsack_mask_mass)
            branch_idx = torch.nonzero(~mass_exist)
            # parent_idx[torch.nonzero(mass_exist)] -= 1
            mass_tag[torch.nonzero(mass_exist)] = diff_mass[torch.nonzero(mass_exist)]
            # print('diff_mass', diff_mass, 'parent_idx', parent_idx, 'mass_tag', mass_tag, 'branch_idx', branch_idx)
            j = 2
            while torch.any(mass_exist) or j < min(5, len(mass_block)-2):
                diff_mass =  mass_block[j:] - mass_block[:-j]
                # TODO: doubled self.knapsack_mask_mass
                # print('mass_exist', mass_block[j:], mass_block[:-j], diff_mass, parent_idx, j)
                mass_exist = self.find_if_list_mass_is_mono(diff_mass[branch_idx-j+1], self.knapsack_mask_mass)
                parent_idx[branch_idx[mass_exist]] = branch_idx[mass_exist]-j+1
                mass_tag[branch_idx[mass_exist]] = diff_mass[branch_idx[mass_exist]-j+1]
                # branch_idx = branch_idx[~mass_exist]
                j+=1
            non_zero_idx = mass_tag != 0
            mass_tags.append(mass_tag[non_zero_idx])
            parent_mass.append(mass_block[parent_idx[non_zero_idx]])
            # print('mass_tag', mass_tag[non_zero_idx])
            # print('parent_idx', mass_block[parent_idx[non_zero_idx]], )
            # print('mass_block', mass_block)
            mass_blocks.append(mass_block)
        # print(stop)
        return mass_tags, parent_mass

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




