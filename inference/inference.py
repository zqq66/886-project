import torch
import math
from dataclasses import dataclass, field
from gnova.utils.BasicClass import Residual_seq, Composition
import operator
from copy import deepcopy
import numpy as np
from gnova.utils.Glycan import Glycan, convert2glycoCT, Monosaccharide
from gnova.data.label_generator import unique_int

@dataclass
class Pep_Inference_Status:
    psm_idx: str
    idx: int
    inference_seq: list[int]
    label_seq: list[str]
    precursor_mass: float
    mass_list: list[float]
    parent_list: list[float]
    parent_mass_list: torch.Tensor
    parent_mono_list: list[float]
    optimal_path: object
    ms1_threshold: float
    ms2_threshold: float
    current_mass: float = 0
    pred_seq_block_index: int = 0
    score_list: list = field(default_factory=list)
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

class Inference_knap(object):
    def __init__(self, cfg, model,model_knapsack, inference_dl, aa_dict, tokenize_aa_dict, detokenize_aa_dict, knapsack_mask):
        self.cfg = cfg
        self.model = model
        self.model_knapsack = model_knapsack
        self.inference_dl_ori = inference_dl
        self.knapsack_mask_mass = knapsack_mask['mass']
        self.knapsack_mask_aa = knapsack_mask['aa_composition']
        self.knapsack_mask_aa = self.knapsack_mask_aa[np.argsort(self.knapsack_mask_mass)]
        self.knapsack_mask_mass = np.sort(self.knapsack_mask_mass)
        self.aa_dict = aa_dict
        self.tokenize_aa_dict = tokenize_aa_dict
        self.detokenize_aa_dict = detokenize_aa_dict
        self.g = Glycan(aa_dict)
        self.canbecorrected = set()
        self.unbale2predict = set()
        self.range_tensor = torch.arange(self.cfg.data.peptide_max_len)[1::2]
        self.range_tensor = self.range_tensor.repeat_interleave(2)
        self.range_tensor = torch.concat((torch.tensor([0]), self.range_tensor))
        self.prediction_rate = []

    def __iter__(self):
        self.inference_dl = iter(self.inference_dl_ori)
        return self

    def __next__(self):
        pos_index = 1
        pep_finish_pool = {}
        tgt, mems_ori, pep_status_list, product_ion_moverz_ori, product_ion_mask_ori, device, psm_idx = self.exploration_initializing()
        pep_status_list, pep_finish_pool = self.next_aa_choice(tgt, pep_status_list, pep_finish_pool)
        tgt, glycan_mass, glycan_crossattn_mass, parent_mono_lists = self.decoder_inference_input_gen(pep_status_list, device)
        mem, product_ion_moverz, product_ion_mask = self.past_input_stacker(pep_status_list, mems_ori, product_ion_moverz_ori, product_ion_mask_ori)

        while len(pep_status_list) > 0:
            true_idx = math.ceil(tgt.shape[-1] / 2)
            # print('tgt.shape', tgt)
            range_tensor = torch.arange(true_idx + 1)
            with torch.inference_mode():
                tgt, _ = self.model.tgt_get(tgt=tgt,
                                            src=mem,
                                            pos_index=self.range_tensor[:tgt.shape[1]].cuda().unsqueeze(0),
                                            glycan_crossattn_mass=glycan_crossattn_mass,
                                            glycan_mass=glycan_mass,
                                            rel_mask=product_ion_mask,
                                            node_mass=product_ion_moverz,
                                            parent_mono_lists=parent_mono_lists)
            pep_status_list, pep_finish_pool = self.next_aa_choice(tgt, pep_status_list, pep_finish_pool)
            if len(pep_status_list) <= 0: break
            tgt, glycan_mass, glycan_crossattn_mass, parent_mono_lists = self.decoder_inference_input_gen(pep_status_list, device)
            mem, product_ion_moverz, product_ion_mask = self.past_input_stacker(pep_status_list, mems_ori,
                                                                                product_ion_moverz_ori,
                                                                                product_ion_mask_ori)
        print('len result', len(pep_finish_pool))
        # print(stop)
        if len(pep_finish_pool) < len(psm_idx):
            able2predict = set(i.psm_idx for i in pep_finish_pool)
            self.unbale2predict.union(set(psm_idx) - able2predict)
        self.prediction_rate.append(len(pep_finish_pool)/len(psm_idx))

        return pep_finish_pool

    def find_if_list_mass_is_mono(self, mass_list, reference):
        ms2_left_boundary = reference.searchsorted(mass_list - 0.04, side='left')
        ms2_right_boundary = reference.searchsorted(mass_list + 0.04, side='right')
        return torch.tensor(ms2_right_boundary>ms2_left_boundary)

    def obtain_optimal_path(self, encoder_input, label):
        # TODO: bug fix
        # todo:if no true regenerate mono type
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

    def exploration_initializing(self):
        encoder_input, decoder_input, seq, precursor_mass, pep_mass, psm_index, charge_threshold,label,label_mask = next(self.inference_dl)
        with torch.inference_mode(): mem = self.model.mem_get(**encoder_input)
        # print('seq', seq, psm_index, encoder_input['node_mass'][label])

        mass_tags, parent_mass = self.obtain_optimal_path(encoder_input, label)
        # print('mass_blocks', mass_blocks)
        pep_status_list = []
        for i in range(len(seq)):
            pool = Pep_Inference_BeamPool(max_size=self.cfg.inference.beam_size)
            pep_status_list.append(pool)
            if self.cfg.inference.ms1_threshold_unit=='ppm':
                ms1_threshold = (precursor_mass[i]+Composition('H2O').mass+\
                                 charge_threshold[i]*Composition('proton').mass) * self.cfg.inference.ms1_threshold * 1e-6
            elif self.cfg.inference.ms1_threshold_unit=='Th': ms1_threshold = self.cfg.inference.ms1_threshold
            else: raise NotImplementedError

            if self.cfg.inference.ms2_threshold_unit=='Th': ms2_threshold = self.cfg.inference.ms2_threshold*2
            else: raise NotImplementedError

            pool.put(Pep_Inference_Status(psm_idx=psm_index[i], idx=i, label_seq=seq[i],
                                          precursor_mass=precursor_mass[i],
                                          ms1_threshold = ms1_threshold,
                                          ms2_threshold = ms2_threshold,
                                          optimal_path =(mass_tags[i], parent_mass[i]),#encoder_input['node_mass'][i][label[i]],
                                          inference_seq=[0], mass_list=[0.], parent_list=[0.], current_mass=0,
                                          parent_mass_list=torch.tensor([0.]), parent_mono_list=[0.]))
        tgt, glycan_mass, glycan_crossattn_mass, parent_mono_lists = self.decoder_inference_input_gen(pep_status_list, mem.device)

        with torch.inference_mode():
            tgt, _ = self.model.tgt_get(tgt=tgt,
                                        src=mem,
                                        pos_index=torch.zeros((tgt.shape[0], 1), dtype=torch.long, device=mem.device),
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
                next_aa = tgt[i,-1]
                if tgt.shape[1] % 2 == 0: # keep only one parent index
                    max_val, max_idx = torch.max(next_aa.unsqueeze(-1), dim=0)
                    next_aa = torch.full_like(next_aa, float('-inf'))
                    next_aa[max_idx] = max_val
                # print('next_aa', next_aa)
                for aa_id in torch.nonzero(~torch.isneginf(
                        next_aa)):
                    # wait until parent index predicted
                    # print('aa_id', aa_id, next_aa[aa_id])
                    is_parent_idx = tgt.shape[1] % 2 == 0
                    # prioritize mono: if knapsack is false instead of trying other index change mono first
                    aa_id = aa_id.item()
                    if is_parent_idx:
                        knapsack_mask = self.knapsack_mask_builder(current_status, aa_id)
                        # print('knapsack_mask', knapsack_mask, aa_id, next_aa[aa_id], self.detokenize_aa_dict[current_status.inference_seq[-1]])
                    else:
                        knapsack_mask = True
                        # print('mono', aa_id, next_aa[aa_id])
                    if knapsack_mask:
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
                        # TODO: mass_list or other list should be able to accumulate after mono assigned to one parent idx
                        # use that as parent mass
                        current_mass = self.detokenize_aa_dict[aa_id] if tgt.shape[1] % 2 != 0 else 0
                        current_status_new.current_mass += current_mass
                        current_status_new.total_score += next_aa[aa_id]
                        # print('seq', current_status_new.inference_seq, 'idx', tgt.shape[1])
                        current_status_new.total_inference_len += 1
                        current_status_new.score = current_status_new.total_score
                        current_status_new.score_list += [next_aa[aa_id]]
                        current_status_new.mass_list = torch.cumsum(mass_list, dim=0)
                        current_status_new.parent_list = mass_list_parent
                        # print('inf seq', current_status_new.inference_seq, tgt[i, -1, :])
                        # print('mass_list', current_status_new.mass_list, current_status_new.parent_list)
                        current_status_new.parent_mono_list = mass_list
                        if ((abs(current_status_new.precursor_mass - current_status_new.current_mass) < 5 or
                            current_status_new.precursor_mass < current_status_new.current_mass) and
                            tgt.shape[1] % 2 == 0) or \
                                len(current_status_new.inference_seq) >= self.cfg.data.peptide_max_len:
                            # print('finish', current_status_new.inference_seq, current_status_new.current_mass, current_status_new.precursor_mass)
                            if current_status_new.idx in pep_finish_pool:
                                if pep_finish_pool[current_status_new.idx].score < current_status_new.score:
                                    pep_finish_pool[current_status_new.idx] = Pep_Finish_Status(
                                        psm_idx=deepcopy(current_status_new.psm_idx),
                                        inference_seq=deepcopy(current_status_new.inference_seq[1:]),
                                        label_seq=deepcopy(current_status_new.label_seq),
                                        score=current_status_new.score,
                                        score_list=deepcopy(current_status_new.score_list))
                            else:
                                pep_finish_pool[current_status_new.idx] = Pep_Finish_Status(
                                    psm_idx=deepcopy(current_status_new.psm_idx),
                                    inference_seq=deepcopy(current_status_new.inference_seq[1:]),
                                    label_seq=deepcopy(current_status_new.label_seq),
                                    score=current_status_new.score,
                                    score_list=deepcopy(current_status_new.score_list))
                        else:
                            correct_lst, gt_seq = self.g.pre2glycan_labels_inversed(
                                torch.tensor(current_status_new.inference_seq), convert2glycoCT(current_status_new.label_seq))
                            if torch.count_nonzero(correct_lst==0) >2:
                                node_matched = current_status_new.optimal_path[1]
                                cur_mass = current_status_new.mass_list[-2]
                                # print('mass compare',current_status_new.precursor_mass, current_status_new.current_mass, cur_mass)
                                # print('current_status_new.inference_seq', current_status_new.inference_seq, correct_lst)

                                # path_mass_match_left = node_matched.searchsorted(
                                #     cur_mass - current_status_new.ms2_threshold)
                                # path_mass_match_right = node_matched.searchsorted(
                                #     cur_mass + current_status_new.ms2_threshold)
                                # if path_mass_match_left < path_mass_match_right:
                                #     self.canbecorrected.add(current_status_new.psm_idx)
                            #     else:
                            #         pool.put(current_status_new)
                            # else:
                            #     pool.put(current_status_new)
                            pool.put(current_status_new)
                    # else:
                    # #     print('mass left', current_status.precursor_mass - current_status.current_mass)
                    # #     print('inference_seq', current_status.inference_seq[1:])
                    #     print('current_status_new.label_seq', current_status.label_seq)

                i+=1
            if len(pool.pool)>0:
                if tgt.shape[1] % 2 == 0:
                    pool.sort()
                new_pep_status_list.append(pool)

        return new_pep_status_list, pep_finish_pool

    def knapsack_mask_builder(self, current_status, p_id):
        # even based on latest mass, the next prediction may not rely on this mass
        # knapsack should be obtained when idx is predicted
        # if the predicted mono doesn't meet the mass find second best prediction
        (mass_tags, parent_masses) = current_status.optimal_path
        # print('no', parent_masses.shape)
        if not parent_masses.shape:
            return True
        path_mass = current_status.mass_list[-2]
        path_mass_match_left = parent_masses.searchsorted(path_mass-current_status.ms2_threshold)
        path_mass_match_right = parent_masses.searchsorted(path_mass+current_status.ms2_threshold)
        mass_tag = (mass_tags[path_mass_match_left:path_mass_match_right]).astype(int)
        predict_mono = int(self.detokenize_aa_dict[current_status.inference_seq[-1]])
        # print('path_mass', path_mass, parent_masses, mass_tag, predict_mono, mass_tag ==predict_mono)
        if (mass_tag == predict_mono).any() or len(mass_tag) < 1:
            # current_status.optimal_path[0][path_mass_match_left:path_mass_match_right][mass_tag == predict_mono] = 0
            return True
        else:return False

    def decoder_inference_input_gen(self, pep_status_list, device):
        tgt = []
        current_mass = []
        current_cross_atte_mass = []
        parent_mono_lists = []

        for current_status_pool in pep_status_list:
            for current_status in current_status_pool.pool:
                # print('current_status.inference_seq', current_status.inference_seq)
                tgt.append(torch.tensor(current_status.inference_seq))
                nterm_mass = torch.tensor(current_status.mass_list).cuda()
                cterm_mass = current_status.precursor_mass - nterm_mass
                parent_mass = torch.tensor(current_status.parent_list).cuda()
                current_mass.append(torch.stack([nterm_mass, cterm_mass], dim=1))
                current_cross_atte_mass.append(torch.stack([nterm_mass, parent_mass], dim=1))
                parent_mono_lists.append(torch.tensor(current_status.parent_mono_list))

                # print('tgt', tgt, current_mass, current_cross_atte_mass)
        tgt = torch.stack(tgt)
        pep_mass = torch.stack(current_mass)
        pep_crossattn_mass = torch.stack(current_cross_atte_mass)
        parent_mono_list = torch.stack(parent_mono_lists).to(device)

        return tgt.cuda(), pep_mass.cuda(), pep_crossattn_mass.cuda(), parent_mono_list

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
