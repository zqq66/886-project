import copy

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from gnova.utils.BasicClass import Residual_seq
import torch.distributed as dist
from gnova.utils.Glycan import Glycan, convert2glycoCT, Monosaccharide
import numpy as np
import math


@dataclass
class Pep_Inference_Status:
    idx: int
    inference_seq: torch.Tensor
    label_seq: Monosaccharide
    gt_seq: torch.Tensor
    correct_lst: torch.Tensor
    # accu_mass_list: list[float]
    # parent_mass_list: list[float]
    precursor_mass: float
    pep_mass: float
    current_aa_index: int = 0
    consecutive_correct_flag: bool = True
    current_mass: float = 0


@dataclass
class Pep_Finish_Status:
    idx: int
    precursor_mass: float
    correct_num: int
    accu_mass_list: list[float]
    parent_mass_list: list[float]
    correct_lst: torch.Tensor
    train_seq: list[str]
    gt_seq: list[str]


class Environment(object):
    def __init__(self, cfg, model, inference_dl, aa_dict, tokenize_aa_dict, detokenize_aa_dict):
        self.cfg = cfg
        self.model = model
        self.inference_dl_ori = inference_dl
        self.g = Glycan(aa_dict)
        # print('aa_dict', aa_dict)
        self.aa_dict = aa_dict
        self.tokenize_aa_dict = tokenize_aa_dict
        self.detokenize_aa_dict = detokenize_aa_dict

    def __iter__(self):
        self.inference_dl = iter(self.inference_dl_ori)
        return self

    def __next__(self):
        pos_index = 1
        pep_finish_pool = []
        mem, tgt, past_tgts, past_mems, pep_status_list, node_mass, node_mass_mask = self.exploration_initializing()
        device = mem.device
        product_ion_mask_inference = node_mass_mask.clone()
        pep_status_list, pep_finish_pool, keepindex = self.next_aa_choice(tgt, pep_status_list,
                                                                          pep_finish_pool)

        while len(pep_status_list) > 0:
            # print('pep_status_list', len(pep_status_list))
            tgt, glycan_mass, glycan_crossattn_mass = self.decoder_inference_input_gen(pep_status_list, device)
            product_ion_mask_inference = product_ion_mask_inference[keepindex]
            past_tgts, past_mems = past_tgts[keepindex], past_mems[keepindex]
            with torch.no_grad():
                if dist.is_initialized():
                    tgt, past_tgts = self.model.module.inference_step(tgt=tgt,
                                                                      pos_index=torch.tensor([pos_index],
                                                                                             dtype=torch.long,
                                                                                             device=device),
                                                                      glycan_crossattn_mass=glycan_crossattn_mass,
                                                                      glycan_mass=glycan_mass,
                                                                      past_tgts=past_tgts,
                                                                      past_mems=past_mems,
                                                                      rel_mask=product_ion_mask_inference)
                else:
                    tgt, past_tgts = self.model.inference_step(tgt=tgt,
                                                               pos_index=torch.tensor([pos_index], dtype=torch.long,
                                                                                      device=device),
                                                               glycan_crossattn_mass=glycan_crossattn_mass,
                                                               glycan_mass=glycan_mass,
                                                               past_tgts=past_tgts,
                                                               past_mems=past_mems,
                                                               rel_mask=product_ion_mask_inference)
            pep_status_list, pep_finish_pool, keepindex = self.next_aa_choice(tgt, pep_status_list, pep_finish_pool)
            # tgt, pep_mass, pep_crossattn_mass = self.decoder_inference_input_gen(pep_status_list, device)
            # product_ion_mask_inference = product_ion_mask_inference[keepindex]
            # past_tgts, past_mems = past_tgts[keepindex], past_mems[keepindex]
            pos_index += 1
        # candidate label should be correct lst we obtained
        tgt, glycan_mass, glycan_crossattn_mass, label, label_mask = self.decoder_train_input_gen(pep_finish_pool,
                                                                                                  pos_index + 1, device)
        # print('after env tgt', tgt)
        true_idx = math.ceil(pos_index / 2)
        # print('pos_index', pos_index, tgt.shape)
        range_tensor = torch.arange(true_idx + 1)
        return ({'src': mem,
                 'tgt': tgt,
                 'pos_index': range_tensor.repeat_interleave(2)[1:pos_index + 2].cuda().unsqueeze(0),
                 'rel_mask': node_mass_mask,
                 'node_mass': node_mass,
                 'glycan_mass': glycan_mass,
                 'glycan_crossattn_mass': glycan_crossattn_mass},
                label, label_mask)

    def exploration_initializing(self):
        # data entry for each spectra
        encoder_input, decoder_input, seq, precursor_mass, pep_mass, _, _, label, label_mask = next(self.inference_dl)
        # print(len(seq))
        # 由于需要自由探索，对decoder input里面每个input只取<bos>
        if dist.is_initialized():
            mem = self.model.module.mem_get(**encoder_input)
        else:
            mem = self.model.mem_get(**encoder_input)
        print(seq)
        gt_seq = torch.zeros((len(self.tokenize_aa_dict) + 1, 1))
        gt_seq[0] = 1
        pep_status_list = [Pep_Inference_Status(idx=i,
                                                label_seq=convert2glycoCT(seq[i]),
                                                pep_mass=pep_mass[i],
                                                precursor_mass=precursor_mass[i],
                                                inference_seq=decoder_input['tgt'][i],
                                                correct_lst=torch.tensor([1]),
                                                gt_seq=gt_seq.T,
                                                # accu_mass_list=[0],
                                                # parent_mass_list=[0]
                                                ) for i in range(len(seq))]
        with torch.no_grad():
            if dist.is_initialized():
                tgt, past_tgts, past_mems = self.model.module.inference_initialize(mem=mem, **decoder_input)
            else:
                tgt, past_tgts, past_mems = self.model.inference_initialize(mem=mem, **decoder_input)
        return mem, tgt, past_tgts, past_mems, pep_status_list, encoder_input['node_mass'], encoder_input[
            'rel_mask']  # , decoder_input['product_ion_moverz'], decoder_input['product_ion_mask']#, candidate_label

    def next_aa_choice(self, tgt, pep_status_list, pep_finish_pool):
        tgt = tgt.float().cpu().numpy()
        # print('next aa tgt', tgt.shape)
        # print('tgt_idx', tgt_idx.shape)
        keep_index = torch.ones(len(pep_status_list), dtype=bool)
        for i, current_status in enumerate(pep_status_list):
            # print('tgt.shape', tgt.shape)
            if tgt.shape[-1] == 1:
                next_aa = tgt[i, -1, :].argmax()
                # print('next_aa before', tgt, next_aa,)
            else:
                epsilon = self.cfg.train.epsilon_fulllen_probability  # /math.sqrt(current_status.correct_num)
                next_aa = tgt[i, -1, :]
                # print('next_aa before', next_aa,)
                max_id = next_aa.argmax()
                # print('max_id', max_id)
                # inf_indices = tgt == -float('inf')
                next_aa = np.ones_like(next_aa) * (1 - epsilon ** (1 / len(current_status.label_seq))) / (
                        len(next_aa) - 1)
                # print('next_aa after', next_aa)
                next_aa[max_id] = epsilon ** (1 / len(current_status.label_seq))
                # print('next_aa max_id', next_aa)
                next_aa[tgt[i, -1, :] == float('-inf')] = 0
                next_aa /= np.sum(next_aa)
                # print('next_aa', next_aa)
                next_aa = np.random.choice(len(next_aa), p=next_aa)
            tgt_current = torch.tensor([next_aa]).cuda()
            # gt = torch.argmax(current_status.gt_seq, dim=-1).cuda()
            # if gt.shape[0] > 1:
            #     gt = gt[:-1]
            # print('gt',gt)
            current_status.inference_seq = torch.cat((current_status.inference_seq, tgt_current))
            # print('current_status.inference_seq', current_status.inference_seq)
            current_status.correct_lst, current_status.gt_seq = self.g.pre2glycan_labels_inversed_random(
                current_status.inference_seq,
                copy.deepcopy(current_status.label_seq))
            # print('current_status.correct_lst', current_status.correct_lst)
            current_status.current_aa_index += 1
            # print('correct_seq', correct_seq)
            # features for each time step should be based on prediction
            mass_list = torch.zeros((len(current_status.inference_seq)))
            mass_list[1::2] = torch.tensor(
                [self.detokenize_aa_dict[i.item()] for i in current_status.inference_seq[1::2]])
            mass_list_parent = torch.cumsum(mass_list, dim=0)
            for idx, mass in enumerate(mass_list):
                if idx % 2 != 0 and idx > 0:  # mono
                    mass_list_parent[idx] = mass
                elif idx > 1:  # parent idx
                    mass_list_parent[idx] = mass_list_parent[current_status.inference_seq[idx] + 1] + \
                                            self.detokenize_aa_dict[current_status.inference_seq[idx - 1].item()]
            current_status.current_mass = [torch.sum(mass_list), mass_list_parent[-1]]
            correct_seq = torch.argmax(current_status.gt_seq, dim=-1)  # [:-1]
            # print('correct_seq', correct_seq)
            correct_mass = torch.tensor(
                [self.detokenize_aa_dict[i.item()] for i in correct_seq[1::2]])
            true_pre_idx = len(current_status.correct_lst)
            if torch.count_nonzero(current_status.correct_lst == 0) != 0:
                true_pre_idx = torch.nonzero(current_status.correct_lst == 0)[0] + 1
            # print('current_status.current_mass',torch.sum(correct_mass), current_status.precursor_mass)
            if (abs(torch.sum(correct_mass) - current_status.precursor_mass) < 0.01 and correct_seq.shape[
                0] % 2 != 0) or \
                    (len(current_status.inference_seq) >= 4 and len(
                        current_status.inference_seq) >= 2 * true_pre_idx) or \
                    len(current_status.inference_seq) >= self.cfg.data.peptide_max_len:
                print('true_pre_idx', true_pre_idx, current_status.correct_lst, current_status.inference_seq)
                gt_seq = current_status.gt_seq[:true_pre_idx, :]
                # print('current_status.gt_seq', gt_seq.shape)
                pep_finish_pool += [Pep_Finish_Status(idx=current_status.idx,
                                                      train_seq=current_status.inference_seq,
                                                      gt_seq=gt_seq,
                                                      precursor_mass=current_status.precursor_mass,
                                                      correct_num=torch.count_nonzero(current_status.correct_lst),
                                                      correct_lst=current_status.correct_lst,
                                                      accu_mass_list=torch.cumsum(mass_list, dim=0),
                                                      parent_mass_list=mass_list_parent)]
                keep_index[i] = False

        new_pep_status_list = []
        for keep_flag, current_status in zip(keep_index, pep_status_list):
            if keep_flag: new_pep_status_list.append(current_status)

        return new_pep_status_list, pep_finish_pool, keep_index

    def decoder_inference_input_gen(self, pep_status_list, device):
        tgt = []
        current_mass = []
        current_cross_mass = []
        for current_status in pep_status_list:
            # TODO teacher forcing: make input for each step guaranteed to be correct
            # print('inference_seq', current_status.inference_seq)
            tgt.append(current_status.inference_seq)
            # print('tgt', tgt, current_status.current_mass)
            current_cross_mass.append(current_status.current_mass)
            current_mass.append(
                [current_status.current_mass[0], current_status.precursor_mass - current_status.current_mass[0]])
        # print('tgt', tgt)
        # print('current_mass', current_mass, 'current_cross_mass', current_cross_mass)
        tgt = torch.stack(tgt)  # .unsqueeze(-1)
        glycan_mass = torch.tensor(current_mass, dtype=torch.float64, device=device).unsqueeze(1)
        glycan_crossattn_mass = torch.tensor(current_cross_mass, dtype=torch.float64, device=device).unsqueeze(
            1)  # torch.concat([glycan_mass/i for i in range(1,self.cfg.model.max_charge+1)],dim=-1)
        # print('tgt, pep_mass, pep_crossattn_mass', tgt.shape, glycan_mass.shape, glycan_crossattn_mass.shape)
        return tgt, glycan_mass, glycan_crossattn_mass

    def decoder_train_input_gen(self, pep_finish_pool, max_len, device):
        tgt, pep_mass, pep_crossattn_mass, label, label_mask, index_list = [], [], [], [], [], []
        for current_status in pep_finish_pool:
            gt = current_status.train_seq
            # print('gt', gt)
            # gt = torch.cat((torch.tensor([len(self.tokenize_aa_dict)]), gt), dim=-1)
            tgt_item = gt.cuda()  # torch.tensor([self.aa_dict[current_status.train_seq[i]] for i in range(len(current_status.train_seq))],dtype=torch.long,device=device)
            # print('tgt_item',tgt_item)
            tgt_item = F.pad(tgt_item, (0, max_len - len(gt)))
            nmass = torch.tensor(current_status.accu_mass_list)
            pmass = torch.tensor(current_status.parent_mass_list)  # current_status.precursor_mass - nmass
            cmass = current_status.precursor_mass - nmass
            # print('nmass', nmass, 'cmass', cmass, 'pmass', pmass)

            pep_mass_item = F.pad(torch.stack([nmass, cmass], dim=-1), (0, 0, 0, max_len - len(gt)))
            pep_crossattn_mass_item = F.pad(torch.stack([nmass, pmass], dim=-1), (0, 0, 0, max_len - len(
                gt)))  # torch.concat([pep_mass_item/i for i in range(1, self.cfg.model.max_charge+1)],dim=-1)
            label_item = current_status.gt_seq[1:, :]
            # label_item = candidate_label[current_status.idx,:current_status.correct_num]
            # print('train label_item', label_item.shape,)
            label_item = F.pad(label_item,
                               (0, max(0, max_len - label_item.size(1)), 0, max(0, max_len - label_item.size(0))))
            print('train label_item', torch.argmax(label_item, dim=-1), )
            label_mask_item = F.pad(torch.ones(len(current_status.train_seq), dtype=bool, device=device),
                                    (0, max_len - len(current_status.train_seq)))
            # print('train label_mask_item', label_mask_item, max_len)
            # only keep those after bos
            # print('label_mask_item', label_mask_item)
            tgt.append(tgt_item)
            pep_mass.append(pep_mass_item)
            pep_crossattn_mass.append(pep_crossattn_mass_item)
            label.append(label_item)
            label_mask.append(label_mask_item)
            index_list.append(current_status.idx)
        # print('len(pep_finish_pool)', len(pep_finish_pool))
        index_list = torch.tensor(index_list, dtype=torch.long).argsort()
        tgt = torch.stack(tgt)[index_list]
        pep_mass = torch.stack(pep_mass)[index_list]
        pep_crossattn_mass = torch.stack(pep_crossattn_mass)[index_list]
        label = torch.stack(label)[index_list]
        label_mask = torch.stack(label_mask)[index_list]

        return tgt.cuda(), pep_mass.cuda(), pep_crossattn_mass.cuda(), label.cuda(), label_mask.cuda()