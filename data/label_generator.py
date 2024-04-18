import torch
import math
import copy
import numpy as np
from gnova.utils.Glycan import Glycan, convert2glycoCT, Monosaccharide
import torch.distributed as dist
import torch.nn.functional as F
import pickle


class LabelGenerator:
    def __init__(self, cfg, model, inference_dl, knapsack_mask, name2mass, name2id):
        self.cfg = cfg
        self.inference_dl_ori = inference_dl
        self.model = model
        self.knapsack_mask_mass = knapsack_mask['mass']
        self.id2mass = {i:m for i, m in enumerate(self.knapsack_mask_mass)}
        self.name2mass = name2mass
        self.name2id = name2id
        self.id2name = {v:k for (k, v) in self.name2id.items()}
        self.knapsack_mask_mass = np.sort(self.knapsack_mask_mass)
        with open(self.cfg.glycan_db, 'rb') as f:
            self.glycan_dict = pickle.load(f)


    def __iter__(self):
        self.inference_dl = iter(self.inference_dl_ori)
        return self

    def __next__(self):
        encoder_input, decoder_input, seq, precursor_mass, pep_mass, psm_index, _, label, label_mask = next(self.inference_dl)
        if dist.is_initialized():
            mem = self.model.module.mem_get(**encoder_input)
        else:
            mem = self.model.mem_get(**encoder_input)
        print('seq', seq)
        tgt, mass_list, pos_indices = self.generate_sequence(encoder_input, label, seq)
        pos_index = max(pos_indices)
        tgt, glycan_mass, glycan_crossattn_mass,parent_mono_lists, tgt_label, tgt_label_mask = \
            self.decoder_input_generator(tgt, mass_list, pos_indices, precursor_mass)
        # true_idx = math.ceil(pos_index / 2)
        range_tensor = torch.arange(pos_index*2)[1::2]
        range_tensor = range_tensor.repeat_interleave(2)
        print('psm_index', psm_index)
        return ({'src': mem,
                 'tgt': tgt,
                 'pos_index': torch.concat((torch.tensor([0]), range_tensor[:-1])).cuda().unsqueeze(0),
                 'rel_mask': encoder_input['rel_mask'],
                 'node_mass': encoder_input['node_mass'],
                 'glycan_mass': glycan_mass,
                 'glycan_crossattn_mass': glycan_crossattn_mass,
                 'parent_mono_lists':parent_mono_lists},
                tgt_label, tgt_label_mask)

    def obtain_parent_mass(self, mass_list, tgt):
        mass_list_parent = torch.cumsum(mass_list, dim=0)
        mass_branch_parent = torch.zeros_like(mass_list)
        for idx, mass in enumerate(mass_list):
            if idx % 2 != 0 and idx > 0:  # mono
                mass_list_parent[idx] = mass
            elif idx > 2:  # parent idx
                mass_list_parent[idx] = mass_list_parent[tgt[idx] + 1] + \
                                        self.id2mass[tgt[idx - 1].item()]

        return mass_list_parent

    def decoder_input_generator(self, tgts, mass_lists, max_lens, precursor_mass):
        padded_tgts, pep_mass, pep_crossattn_mass, parent_mono_lists,label, label_mask = [], [], [], [], [],[]
        max_len_batch = max(max_lens)*2
        for i,tgt in enumerate(tgts):
            # print('gt', tgt)
            max_len = max_lens[i]*2
            # gt = torch.cat((torch.tensor([len(self.tokenize_aa_dict)]), gt), dim=-1)
            tgt_item = tgt.cuda()  # torch.tensor([self.aa_dict[current_status.train_seq[i]] for i in range(len(current_status.train_seq))],dtype=torch.long,device=device)
            # print('tgt_item',tgt_item)
            tgt_item = F.pad(tgt_item, (0, max_len_batch - len(tgt)))
            nmass = torch.cumsum(mass_lists[i], dim=0)
            pmass = self.obtain_parent_mass(mass_lists[i], tgt)  # current_status.precursor_mass - nmass
            cmass = precursor_mass[i] - nmass
            pep_mass_item = F.pad(torch.stack([nmass, cmass], dim=-1), (0, 0, 0, max_len_batch - len(tgt)))
            pep_crossattn_mass_item = F.pad(torch.stack([nmass, pmass], dim=-1), (0, 0, 0, max_len_batch - len(
                tgt)))  # torch.concat([pep_mass_item/i for i in range(1, self.cfg.model.max_charge+1)],dim=-1)
            parent_mono_list = F.pad(mass_lists[i], ( 0, max_len_batch - len(tgt)))
            # print('pep_mass_item', pep_mass_item, pep_crossattn_mass_item, )
            # print('pep_crossattn_mass_item', pep_crossattn_mass_item.transpose(0, 1))
            label_item = F.one_hot(tgt[1:max_len+1], num_classes=max_len)
            # label_item = candidate_label[current_status.idx,:current_status.correct_num]
            label_item = F.pad(label_item,
                               (0, max(0, max_len_batch - max_len), 0, max(0, max_len_batch - max_len)))
            label_mask_item = F.pad(torch.ones(max_len, dtype=bool),
                                    (0, max_len_batch - max_len))
            # only keep those after bos
            # print('label_mask_item',label_item, label_mask_item.shape, max_len_batch, max_len)
            padded_tgts.append(tgt_item)
            pep_mass.append(pep_mass_item)
            pep_crossattn_mass.append(pep_crossattn_mass_item)
            parent_mono_lists.append(parent_mono_list)
            label.append(label_item)
            label_mask.append(label_mask_item)

        # print('len(pep_finish_pool)', len(pep_finish_pool))
        tgt = torch.stack(padded_tgts)
        pep_mass = torch.stack(pep_mass)
        pep_crossattn_mass = torch.stack(pep_crossattn_mass)
        label = torch.stack(label)
        label_mask = torch.stack(label_mask)
        parent_mono_lists = torch.stack(parent_mono_lists)
        return tgt.cuda(), pep_mass.cuda(), pep_crossattn_mass.cuda(), parent_mono_lists.cuda(), label.cuda(), label_mask.cuda()


    def find_if_list_mass_is_mono(self, mass_list, reference):
        ms2_left_boundary = reference.searchsorted(mass_list - 0.04, side='left')
        ms2_right_boundary = reference.searchsorted(mass_list + 0.04, side='right')
        return torch.tensor(ms2_right_boundary>ms2_left_boundary)


    def obtain_optimal_path(self, encoder_input, label):
        mass_tags = []
        parent_mass = []
        mass_blocks = []
        min_mono = min(self.knapsack_mask_mass)
        for i in range(label.shape[0]):
            mass_block = (encoder_input['node_mass'][i])[label[i]>0]
            mass_block = mass_block[mass_block >= min_mono].cpu()
            mass_block = unique_int(mass_block).numpy()
            print('mass_block', mass_block)
            mass_block = np.append(0, mass_block)
            parent_idx = torch.arange(mass_block.shape[0])
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
                mass_exist = self.find_if_list_mass_is_mono(diff_mass[branch_idx - j + 1], self.knapsack_mask_mass)
                parent_idx[branch_idx[mass_exist]] = branch_idx[mass_exist] - j + 1
                mass_tag[branch_idx[mass_exist]] = diff_mass[branch_idx[mass_exist] - j + 1]
                # mass_exist_ori = torch.logical_or(mass_exist_ori, mass_exist)
                branch_idx = branch_idx[torch.where(torch.logical_and(~mass_exist, branch_idx > j - 1))[0]]
                j += 1
            non_zero_idx = mass_tag != 0
            mass_tags.append(mass_tag[non_zero_idx])
            parent_mass.append(mass_block[parent_idx[non_zero_idx]])
            mass_blocks.append(mass_block)
        return mass_tags, parent_mass

    def generate_sequence(self, encoder_input, labels, seqs):
        mass_tags, parent_masses = self.obtain_optimal_path(encoder_input, labels)
        gts = []
        mass_lists = []
        pos_indices = []
        for idx, seq in enumerate(seqs):
            root = convert2glycoCT(seq)
            root.set_index(0)
            mass_tag, parent_mass = mass_tags[idx],parent_masses[idx]
            # print(' mass_tag, parent_mass',  mass_tag, parent_mass)
            unassigned_node = copy.copy(root.children)
            gt = torch.zeros(self.cfg.data.peptide_max_len, dtype=torch.int64)
            mass_list = torch.zeros(self.cfg.data.peptide_max_len)
            pos_index = 0
            if isinstance(parent_mass, np.floating) :
                parent_mass = [parent_mass]
            # print(seq, mass_tag, parent_mass)
            for i, (m, p) in enumerate(zip(mass_tag, parent_mass)):
                m = int(m)
                unassigned_node = [i for i in unassigned_node if i.index is None]
                unassigned_node_mass = [int(self.name2mass[r.name]) for r in unassigned_node]

                if m in unassigned_node_mass:
                    node_idx = unassigned_node_mass.index(m)
                    new_node = unassigned_node.pop(node_idx)
                    new_node.set_index(pos_index*2+1)
                    new_node_id = self.name2id[new_node.name]
                    gt[new_node.index] = new_node_id
                    mass_list[new_node.index] = self.name2mass[new_node.name]
                    gt[new_node.index+1] = new_node.parent.index
                    unassigned_node += new_node.children if new_node.children is not None else []
                    pos_index+=1
                else:
                    if (sum(mass_list) - p) > int(min(self.name2mass.values())):
                        continue
                    m_idx = [int(i) for i in self.name2mass.values()].index(m)
                    possible_mono = root.find_first_unassigned_mono_with_mf(self.id2name[m_idx])
                    if possible_mono is None:
                        continue
                    parent_mono = copy.copy(possible_mono)
                    while parent_mono.parent.index is None:
                        parent_mono = parent_mono.parent
                    parent_monos = copy.copy(parent_mono.parent.children)
                    parent_mono = parent_monos.pop(0)
                    while parent_mono != possible_mono:
                        parent_mono.set_index(pos_index * 2 + 1)
                        gt[parent_mono.index] = self.name2id[parent_mono.name]
                        mass_list[parent_mono.index] = self.name2mass[parent_mono.name]
                        gt[parent_mono.index + 1] = parent_mono.parent.index
                        parent_monos += copy.copy(parent_mono.children) if parent_mono.children is not None else []
                        parent_mono = parent_monos.pop(0)
                        pos_index += 1
                    possible_mono.set_index(pos_index * 2 + 1)
                    gt[possible_mono.index] = self.name2id[possible_mono.name]
                    mass_list[possible_mono.index] = self.name2mass[possible_mono.name]
                    gt[possible_mono.index + 1] = possible_mono.parent.index
                    parent_monos += copy.copy(possible_mono.children) if possible_mono.children is not None else []
                    pos_index += 1
                    unassigned_node += parent_monos
                    unassigned_node += copy.copy(possible_mono.children) if possible_mono.children is not None else []

            unassigned_node = [i for i in unassigned_node if i.index is None]
            while unassigned_node:
                new_node = unassigned_node.pop(0)
                new_node.set_index(pos_index * 2 + 1)
                new_node_id = self.name2id[new_node.name]
                gt[new_node.index] = new_node_id
                mass_list[new_node.index] = self.name2mass[new_node.name]
                gt[new_node.index + 1] = new_node.parent.index
                unassigned_node += copy.copy(new_node.children) if new_node.children is not None else []
                unassigned_node = [i for i in unassigned_node if i.index is None]
                pos_index += 1
            gts.append(gt)
            mass_lists.append(mass_list)
            pos_indices.append(pos_index)
        return gts, mass_lists, pos_indices

def read_from_db(g_id, glycan_dict):
    glycan = glycan_dict[str(g_id)]['GLYCAN'].clone()
    return glycan.root

def unique_int(mass_block):
    mass_block_int = torch.floor(mass_block)
    unique_indices = {}
    for idx, int_part in enumerate(mass_block_int):
        if int_part.item() not in unique_indices:
            unique_indices[int_part.item()] = idx

    # Now create a mask to filter out the unique values.
    mask = torch.zeros(len(mass_block), dtype=torch.bool)
    for idx in unique_indices.values():
        mask[idx] = True
    unique_float_tensor = mass_block[mask]

    # print('unique_float_tensor', unique_float_tensor)
    # mass_block = torch.tensor([0., 203.09794617, 406.19232178, 568.21813965, 892.3137207, 1095.40724181])
    return unique_float_tensor

