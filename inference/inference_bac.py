import torch
from dataclasses import dataclass, field
from gnova.utils.BasicClass import Residual_seq, Composition
import operator
from copy import deepcopy
import math
import numpy as np

@dataclass
class Pep_Inference_Status:
    psm_idx: str
    idx: int
    inference_seq: list[int]
    label_seq: list[str]
    mass_list: list[float]
    parent_list: list[float]
    parent_mass_list: torch.Tensor
    parent_mono_list: list[float]
    precursor_mass: float
    ms1_threshold: float
    ms2_threshold: float
    peptide_mass: float
    score_list: list = field(default_factory=list)
    current_mass: float = 0
    score: float = 0
    total_score: float = 0
    total_inference_len: float = 0

@dataclass
class Pep_Finish_Status:
    psm_idx: str
    inference_seq: str
    label_seq: str
    score_list: list
    score: float

class Pep_Inference_BeamPool(object):
    def __init__(self, max_size):
        self.pool = []
        self.max_size = max_size

    def is_empty(self): return len(self.pool) == 0

    def put(self, data):
        self.pool.append(data)

    def get(self):
        return self.pool.pop(0)

    def sort(self):
        self.pool = sorted(self.pool,
                           key=operator.attrgetter('score'),
                           reverse=True)[:self.max_size]

class Inference(object):
    def __init__(self, cfg, model, inference_dl, aa_dict, tokenize_aa_dict, detokenize_aa_dict, knapsack_mask):
        self.cfg = cfg
        self.model = model
        self.inference_dl_ori = inference_dl
        self.knapsack_mask_mass = knapsack_mask['mass']
        self.knapsack_mask_aa = knapsack_mask['aa_composition']

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
        pep_finish_pool = {}
        tgt, mems_ori, pep_status_list, product_ion_moverz_ori, product_ion_mask_ori, device, psm_index = self.exploration_initializing()

        pep_status_list, pep_finish_pool = self.next_aa_choice(tgt, pep_status_list, pep_finish_pool)
        tgt, glycan_mass, glycan_crossattn_mass,parent_mono_lists = self.decoder_inference_input_gen(pep_status_list, device)
        mem, product_ion_moverz, product_ion_mask = self.past_input_stacker(pep_status_list, mems_ori, product_ion_moverz_ori, product_ion_mask_ori)

        while len(pep_status_list)>0:

            with torch.inference_mode():
                tgt, _ = self.model.tgt_get(tgt=tgt,
                                         src=mem,
                                         pos_index=self.range_tensor[:tgt.shape[1]].cuda().unsqueeze(0),
                                         glycan_crossattn_mass=glycan_crossattn_mass,
                                         glycan_mass=glycan_mass,
                                         rel_mask=product_ion_mask,
                                        parent_mono_lists=parent_mono_lists,
                                         node_mass=product_ion_moverz)
            pep_status_list, pep_finish_pool = self.next_aa_choice(tgt, pep_status_list, pep_finish_pool)
            if len(pep_status_list)<=0: break
            tgt, glycan_mass, glycan_crossattn_mass,parent_mono_lists = self.decoder_inference_input_gen(pep_status_list, device)
            mem, product_ion_moverz, product_ion_mask = self.past_input_stacker(pep_status_list, mems_ori, product_ion_moverz_ori, product_ion_mask_ori)
        if len(pep_finish_pool) < len(psm_index):
            print('psm_index', set(psm_index)-set(i.psm_idx for i in pep_finish_pool.values()))
            self.unable2predict= self.unable2predict.union(set(psm_index)-set(i.psm_idx for i in pep_finish_pool.values()))
        self.prediction_rate.append(len(pep_finish_pool)/len(psm_index))
        return pep_finish_pool

    def exploration_initializing(self):
        encoder_input, decoder_input, seq, precursor_mass, pep_mass, psm_index, charge_threshold, label, label_mask = next(self.inference_dl)

        # precursor_mass = precursor_mass.cuda()
        # pep_mass = pep_mass.cuda()
        with torch.inference_mode(): mem = self.model.mem_get(**encoder_input)
        pep_status_list = []
        for i in range(len(seq)):
            mass_block = (encoder_input['node_mass'][i])[label[i] > 0]
            print('mass_block', mass_block, seq[i])
            pool = Pep_Inference_BeamPool(max_size=self.cfg.inference.beam_size)
            pep_status_list.append(pool)

            if self.cfg.inference.ms1_threshold_unit=='ppm':
                ms1_threshold = (precursor_mass[i]+Composition('H2O').mass+\
                                 charge_threshold[i]*Composition('proton').mass) * self.cfg.inference.ms1_threshold * 1e-6
            elif self.cfg.inference.ms1_threshold_unit=='Th': ms1_threshold = self.cfg.inference.ms1_threshold
            else: raise NotImplementedError

            if self.cfg.inference.ms2_threshold_unit=='Th': ms2_threshold = self.cfg.inference.ms2_threshold*2
            else: raise NotImplementedError

            pool.put(Pep_Inference_Status(psm_idx=psm_index[i],idx=i, label_seq=seq[i],
                                          precursor_mass=precursor_mass[i],
                                          peptide_mass = pep_mass[i],
                                          ms1_threshold = ms1_threshold,
                                          ms2_threshold = ms2_threshold,
                                          inference_seq=[0], mass_list=[0.],
                                          parent_list=[0.], current_mass=0,
                                          parent_mass_list=torch.tensor([0.]),
                                          parent_mono_list=[0.]))
        tgt, glycan_mass, glycan_crossattn_mass, parent_mono_lists = self.decoder_inference_input_gen(pep_status_list, mem.device)
        # print('tgt', tgt)
        with torch.inference_mode():
            tgt, _ = self.model.tgt_get(tgt=tgt,
                                     src=mem,
                                     pos_index=self.range_tensor[:tgt.shape[1]].cuda().unsqueeze(0),
                                     rel_mask=encoder_input['rel_mask'],
                                     node_mass=encoder_input['node_mass'],
                                     glycan_mass=glycan_mass,
                                     glycan_crossattn_mass=glycan_crossattn_mass,
                                        parent_mono_lists=parent_mono_lists)

        return tgt, mem, pep_status_list, encoder_input['node_mass'].clone(), encoder_input['rel_mask'].clone(), mem.device, psm_index

    def next_aa_choice(self, tgt, pep_status_list, pep_finish_pool):
        i = 0
        new_pep_status_list = []
        tgt = tgt.float()
        for current_status_pool in pep_status_list:
            pool = Pep_Inference_BeamPool(max_size=current_status_pool.max_size)
            for current_status in current_status_pool.pool:
                next_aa = tgt[i, -1]
                # print(tgt, tgt.shape)
                # print('next_aa',current_status.parent_list, current_status.mass_list, next_aa, tgt.shape)
                knapsack_mask = self.knapsack_mask_builder(current_status)
                # print(torch.nonzero(~torch.isneginf(next_aa)))
                for aa_id in torch.nonzero(~torch.isneginf(
                        next_aa)):  # , knapsack_mask_flag in zip(list(self.detokenize_aa_dict.keys())[:-1], knapsack_mask):
                    # print('next_aa', next_aa[aa_id])
                # aa_id = torch.argmax(next_aa).item()
                    aa_id = aa_id.item()
                    if tgt.shape[1] % 2 != 0:
                        # aa_id = torch.argmax(next_aa).item()
                        knapsack_mask_flag = knapsack_mask[aa_id]
                    else:
                        # aa_id = torch.argmax(next_aa[1:]).item()+1
                        knapsack_mask_flag = True
                    # print('knapsack_mask', knapsack_mask)
                    if knapsack_mask_flag:
                        current_status_new = deepcopy(current_status)
                        current_status_new.inference_seq.append(aa_id)
                        parent_mass = 0
                        mass_list = torch.zeros((len(current_status_new.inference_seq)))
                        mass_list[1::2] = torch.tensor(
                            [self.detokenize_aa_dict[i] for i in current_status_new.inference_seq[1::2]])
                        mass_list_parent = torch.cumsum(mass_list, dim=0)
                        for idx, mass in enumerate(mass_list):
                            if idx % 2 != 0 and idx > 0:  # mono
                                mass_list_parent[idx] = mass
                            elif idx > 2:  # parent idx
                                mass_list_parent[idx] = mass_list_parent[current_status_new.inference_seq[idx] + 1] + \
                                                        self.detokenize_aa_dict[
                                                            current_status_new.inference_seq[idx - 1]]
                        #TODO: mass_list or other list should be able to accumulate after mono assigned to one parent idx
                        #use that as parent mass
                        current_mass = self.detokenize_aa_dict[aa_id] if tgt.shape[1] % 2 != 0 else 0
                        current_status_new.current_mass += current_mass
                        current_status_new.total_score += next_aa[aa_id]
                        print('seq', current_status_new.inference_seq, 'idx', tgt.shape[1])
                        current_status_new.total_inference_len+=1
                        current_status_new.score = current_status_new.total_score
                        current_status_new.score_list += [next_aa[aa_id]]
                        current_status_new.mass_list=torch.cumsum(mass_list, dim=0)
                        current_status_new.parent_list=mass_list_parent
                        # print('inf seq', current_status_new.inference_seq, torch.argmax(tgt[i, :], dim=-1))
                        # print('mass_list', current_status_new.mass_list, current_status_new.parent_list)
                        current_status_new.parent_mono_list=mass_list
                        # # print('mass_list', current_status_new.mass_list, 'parent_list',current_status_new.parent_list, current_status_new.peptide_mass)
                        if (abs(current_status_new.precursor_mass-current_status_new.current_mass)<5 and tgt.shape[1]%2==0) or \
                            len(current_status_new.inference_seq)>=self.cfg.data.peptide_max_len:
                            # print('finish', current_status_new.inference_seq)
                            if current_status_new.idx in pep_finish_pool:
                                if pep_finish_pool[current_status_new.idx].score<current_status_new.score:
                                    pep_finish_pool[current_status_new.idx] = Pep_Finish_Status(psm_idx=deepcopy(current_status_new.psm_idx),
                                                                                                inference_seq=deepcopy(current_status_new.inference_seq[1:]),
                                                                                                label_seq=deepcopy(current_status_new.label_seq),
                                                                                                score=current_status_new.score,
                                                                                                score_list=deepcopy(current_status_new.score_list))
                            else:
                                pep_finish_pool[current_status_new.idx] = Pep_Finish_Status(psm_idx=deepcopy(current_status_new.psm_idx),
                                                                                            inference_seq=deepcopy(current_status_new.inference_seq[1:]),
                                                                                            label_seq=deepcopy(current_status_new.label_seq),
                                                                                            score=current_status_new.score,
                                                                                            score_list=deepcopy(current_status_new.score_list))
                        else:
                            pool.put(current_status_new)

                i+=1
            if len(pool.pool)>0:
                pool.sort()
                new_pep_status_list.append(pool)

        return new_pep_status_list, pep_finish_pool

    def knapsack_mask_builder(self, current_status):
        retain_mass = current_status.precursor_mass-current_status.current_mass

        if retain_mass-current_status.ms1_threshold<self.knapsack_mask_mass.max():
            # only keep the one closest value
            # want to also consider the case with 146*2 = 292
            # knapsack_mask_mass need to be sorted
            # knapsack_mask_mass = np.append(self.knapsack_mask_mass, self.knapsack_mask_mass[-1]*2)
            # knapsack_mask_aa = np.append(self.knapsack_mask_aa, self.knapsack_mask_aa[-1])
            knapsack_mask_aa = self.knapsack_mask_aa[np.argsort(self.knapsack_mask_mass)]
            knapsack_mask_mass = np.sort(self.knapsack_mask_mass)
            # print(knapsack_mask_mass, retain_mass-current_status.ms1_threshold, retain_mass+current_status.ms1_threshold)
            ms1_left_boundary = knapsack_mask_mass.searchsorted(retain_mass-current_status.ms1_threshold)
            ms1_right_boundary = knapsack_mask_mass.searchsorted(retain_mass+current_status.ms1_threshold)
            ms1_knapsack_mask = set(self.detokenize_aa_dict[i] for i in knapsack_mask_aa[ms1_left_boundary:ms1_right_boundary].tolist())
        else:
            ms1_knapsack_mask = set(self.detokenize_aa_dict.values())

        knapsack_mask = [True if aa in ms1_knapsack_mask else False for aa in self.detokenize_aa_dict.values()]
        # if len(ms1_knapsack_mask) != len(set(self.detokenize_aa_dict.values())):
        #     print('knapsack_mask_aa', knapsack_mask_aa, 'self.detokenize_aa_dict',knapsack_mask_mass, current_status.ms1_threshold)
        #     print('ms1_left_boundary', ms1_left_boundary, 'ms1_right_boundary', ms1_right_boundary)
        #     print(retain_mass, 'ms1_knapsack_mask', ms1_knapsack_mask, knapsack_mask, self.detokenize_aa_dict.values())
        return knapsack_mask

    def decoder_inference_input_gen(self, pep_status_list, device):
        tgt = []
        current_mass = []
        current_cross_atte_mass = []
        parent_mono_lists = []
        for current_status_pool in pep_status_list:
            for current_status in current_status_pool.pool:
                # print('current_status.inference_seq', current_status.inference_seq)
                tgt.append(torch.tensor(current_status.inference_seq))
                nterm_mass = torch.tensor(current_status.mass_list)
                cterm_mass = current_status.precursor_mass - nterm_mass
                parent_mass = torch.tensor(current_status.parent_list)
                current_mass.append(torch.stack([nterm_mass,cterm_mass],dim=1))
                current_cross_atte_mass.append(torch.stack([nterm_mass,parent_mass],dim=1))
                parent_mono_lists.append(torch.tensor(current_status.parent_mono_list))
                # print('parent',parent_mass,current_status.parent_mono_list)

                # print('tgt', tgt, current_mass, current_cross_atte_mass)
        tgt = torch.stack(tgt)
        pep_mass = torch.stack(current_mass)
        pep_crossattn_mass = torch.stack(current_cross_atte_mass)
        parent_mono_list = torch.stack(parent_mono_lists).to(device)

        return tgt.cuda(), pep_mass.cuda(), pep_crossattn_mass.cuda(), parent_mono_list

    def tonkenize(self, inference_seq):
        tgt = torch.LongTensor([self.aa_dict['<bos>']]+[self.aa_dict[aa] for aa in inference_seq[:-1]])
        return tgt

    def past_input_stacker(self, pep_status_list, mems_ori, product_ion_moverz_ori, product_ion_mask_ori):
        idx_list = []
        for current_status_pool in pep_status_list:
            for current_status in current_status_pool.pool:
                idx_list.append(current_status.idx)
        idx_list = torch.tensor(idx_list,dtype=torch.long,device=product_ion_mask_ori.device)
        mem = mems_ori.index_select(0,idx_list)
        product_ion_mask = product_ion_mask_ori.index_select(0,idx_list)
        product_ion_moverz = product_ion_moverz_ori.index_select(0,idx_list)
        return mem, product_ion_moverz, product_ion_mask