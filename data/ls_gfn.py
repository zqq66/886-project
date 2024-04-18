import copy
import wandb
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from gnova.utils.BasicClass import Residual_seq
import torch.distributed as dist
from .environment import Environment, Pep_Finish_Status
from gnova.utils.Glycan import Glycan, convert2glycoCT, Monosaccharide
import numpy as np
import math


@dataclass
class Pep_Inference_Status:
    idx: int
    inference_seq: torch.Tensor
    back_inf_seq: torch.Tensor
    fwd_inf_seq: torch.Tensor
    label_seq: Monosaccharide
    gt_seq: torch.Tensor
    correct_lst: torch.Tensor
    precursor_mass: float
    pep_mass: float
    current_aa_index: int = 0
    consecutive_correct_flag: bool = True
    current_mass: float = 0


class Sampling:
    def __init__(self, cfg, fwd_model, back_model, inference_dl, aa_dict, tokenize_aa_dict, detokenize_aa_dict):
        self.cfg = cfg
        self.fwd_model = fwd_model
        self.back_model = back_model
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
        finish_pep_status_list = []
        final_node_mass = []
        final_mem = []
        final_rel_mask = []
        mem, tgt, past_tgts, past_mems, pep_status_list, node_mass, node_mass_mask = self.exploration_initializing()
        device = mem.device
        product_ion_mask_inference = node_mass_mask.clone()
        pep_status_list, finish_pep_status_list, keepindex = self.next_aa_choice(tgt, pep_status_list,
                                                                                 finish_pep_status_list)

        while len(pep_status_list) > 0:
            # print('pep_status_list', len(pep_status_list))
            tgt, glycan_mass, glycan_crossattn_mass, _ = self.decoder_inference_input_gen(pep_status_list, device)
            product_ion_mask_inference = product_ion_mask_inference[keepindex]
            past_tgts, past_mems = past_tgts[keepindex], past_mems[keepindex]
            with torch.no_grad():
                if dist.is_initialized():
                    tgt, past_tgts = self.fwd_model.module.inference_step(tgt=tgt,
                                                                          pos_index=torch.tensor([pos_index],
                                                                                                 dtype=torch.long,
                                                                                                 device=device),
                                                                          glycan_crossattn_mass=glycan_crossattn_mass,
                                                                          glycan_mass=glycan_mass,
                                                                          past_tgts=past_tgts,
                                                                          past_mems=past_mems,
                                                                          rel_mask=product_ion_mask_inference)
                else:
                    tgt, past_tgts = self.fwd_model.inference_step(tgt=tgt,
                                                                   pos_index=torch.tensor([pos_index], dtype=torch.long,
                                                                                          device=device),
                                                                   glycan_crossattn_mass=glycan_crossattn_mass,
                                                                   glycan_mass=glycan_mass,
                                                                   past_tgts=past_tgts,
                                                                   past_mems=past_mems,
                                                                   rel_mask=product_ion_mask_inference)
            pep_status_list, finish_pep_status_list, keepindex = self.next_aa_choice(tgt, pep_status_list,
                                                                                     finish_pep_status_list)
            for idx, (keep_flag, m) in enumerate(zip(keepindex, mem)):
                if not keep_flag:
                    final_node_mass.append(node_mass[idx])
                    final_mem.append(m)
                    final_rel_mask.append(product_ion_mask_inference[idx])
            pos_index += 1
        # TODO: remove duplicates
        sampled_status_list = self.local_sample(finish_pep_status_list, torch.stack(final_node_mass),
                                                torch.stack(final_mem), torch.stack(final_rel_mask), device, pos_index,
                                                self.cfg.train.sample_iteration, self.cfg.train.sample_step)
        fwd_input, back_input, label, label_mask, index_list = self.decoder_train_input_gen(sampled_status_list, pos_index + 1, device)

        # print('after env tgt', tgt)
        true_idx = math.ceil(pos_index / 2)
        # print('pos_index', pos_index, tgt.shape)
        range_tensor = torch.arange(true_idx + 1)
        fwd_input['src'] = mem[index_list]
        fwd_input['pos_index'] = range_tensor.repeat_interleave(2)[1:pos_index + 2].cuda().unsqueeze(0)
        fwd_input['rel_mask'] = node_mass_mask[index_list]
        fwd_input['node_mass'] = node_mass[index_list]

        back_input['src'] = mem[index_list]
        back_input['pos_index'] = range_tensor.repeat_interleave(2)[1:pos_index + 2].cuda().unsqueeze(0)
        back_input['rel_mask'] = node_mass_mask[index_list]
        back_input['node_mass'] = node_mass[index_list]
        # print('label, label_mask', label, label_mask)
        return (fwd_input, back_input ,label, label_mask)

    def obtain_finish_samples(self, pep_status, correct_lst, gt_seq):
        # true_pre_idx = len(correct_lst)
        # if torch.count_nonzero(correct_lst == 0) != 0:
        #     true_pre_idx = torch.nonzero(correct_lst == 0)[0] + 1
        # print('true_pre_idx', true_pre_idx)
        fwd_inf_seq = torch.argmax(gt_seq, dim=-1)[:-1]
        print('fwd_inf_seq', fwd_inf_seq)
        fwd_mass_list, fwd_mass_list_parent = self.obtain_fwd_mass(fwd_inf_seq)
        reversed_seq = copy.deepcopy(fwd_inf_seq)
        reversed_seq[1::2] = fwd_inf_seq[2::2]
        reversed_seq[2::2] = fwd_inf_seq[1::2]
        back_mass_list, back_mass_list_parent = self.obtain_back_mass(reversed_seq)
        # print('fwd_mass_list', fwd_mass_list.shape, 'back_mass_list', back_mass_list.shape)

        return [Pep_Finish_Status(idx=pep_status.idx,
                                                      train_seq=fwd_inf_seq,
                                                      gt_seq=gt_seq,
                                                      precursor_mass=pep_status.precursor_mass,
                                                      correct_num=torch.count_nonzero(correct_lst),
                                                      correct_lst=correct_lst,
                                                      accu_mass_list=torch.cumsum(fwd_mass_list, dim=0),
                                                      parent_mass_list=fwd_mass_list_parent,
                                                      ),
                                    Pep_Finish_Status(idx=pep_status.idx,
                                                      train_seq=reversed_seq,
                                                      gt_seq=gt_seq,
                                                      precursor_mass=pep_status.precursor_mass,
                                                      correct_num=torch.count_nonzero(correct_lst),
                                                      correct_lst=correct_lst,
                                                      accu_mass_list=torch.cumsum(back_mass_list, dim=0),
                                                      parent_mass_list=back_mass_list_parent,
                                                      )
                                    ]

    def local_sample(self, finish_pep_status_list, final_node_mass, final_mem, final_rel_mask, device, pos_index, i,
                     k):

        update_success_rates = []
        sampled_status_list = []
        keep_track_duplication = dict()
        # prediction_len = []
        for idx, pep_status in enumerate(finish_pep_status_list):
            if pep_status.idx in keep_track_duplication.keys():
                keep_track_duplication[pep_status.idx].append(str(pep_status.inference_seq.tolist()))
            else:
                keep_track_duplication[pep_status.idx] = [str(pep_status.inference_seq.tolist())]
            correct_lst, gt_seq = pep_status.correct_lst, pep_status.gt_seq
            sampled_status_list.append(self.obtain_finish_samples(pep_status, correct_lst, gt_seq))
            # prediction_len.append(len(pep_status.inference_seq)>k*2+1)
        # if not all(prediction_len):
        #     return sampled_status_list
        print('Local Search...')
        for _ in range(i):
            updates = []
            # Construct new complete trajectories via deconstruction / reconstruction
            back_forth_sampled = self.backforth_sample(finish_pep_status_list,
                                                        final_node_mass.detach(),
                                                        final_mem.detach(),
                                                        final_rel_mask.detach(),
                                                        device, pos_index, k)
            for idx, pep_status in enumerate(back_forth_sampled):
                # remove duplications
                if str(pep_status.fwd_inf_seq.tolist()) not in keep_track_duplication[pep_status.idx]:
                    keep_track_duplication[pep_status.idx].append(str(pep_status.fwd_inf_seq.tolist()))
                else:
                    continue
                correct_lst, gt_seq = self.g.pre2glycan_labels_inversed_random(
                    pep_status.fwd_inf_seq,
                    copy.deepcopy(pep_status.label_seq))
                print('sampled', pep_status.fwd_inf_seq, pep_status.inference_seq)
                sampled_status_list.append(self.obtain_finish_samples(pep_status, correct_lst, gt_seq))
                # Filtering
                new_reward = torch.count_nonzero(correct_lst)
                ori_reward = torch.count_nonzero(pep_status.correct_lst)
                print('new_reward', new_reward, 'ori_reward', ori_reward)
                lp_update = new_reward - ori_reward
                # Deterministic Filtering
                # if reward is higher: accept
                # if first non_zero reward moved backward: accept
                update = (lp_update > 0).float() or torch.nonzero(correct_lst)[0] >torch.nonzero(pep_status.correct_lst)[0]
                # print('update', update)
                updates.append(update.item())
                if update == 1:
                    # change the sampling trajectory with the higher reward,
                    # make the next iteration of sampling on trajectories with higher reward
                    finish_pep_status_list[idx].inference_seq = finish_pep_status_list[idx].fwd_inf_seq
                # else:
                #     finish_pep_status_list.pop(idx)
                update_success_rate = sum(updates) / len(updates)
                update_success_rates.append(update_success_rate)

        update_success_rates = np.mean(update_success_rates)
        print(f'Update Success Rate: {update_success_rates:.2f}')
        # wandb.log({'Update Success Rate': update_success_rates})
        # print('keep_track_duplication', keep_track_duplication)
        return sampled_status_list

    def epsilon_sampling(self, tgt):
        # print('tgt', tgt)
        tgt = tgt[~torch.isinf(tgt).all(dim=1),:].numpy()
        seq_len = tgt.shape[0]
        # print('tgt', tgt)
        # print('seq_len',seq_len )
        # epsilon = self.cfg.train.epsilon_fulllen_probability  # /math.sqrt(current_status.correct_num)

        next_aa = tgt[-1, :]
        # print('next_aa before', next_aa,)
        max_id = next_aa.argmax()
        # print('next_aa argmax', max_id)

        # print('max_id', max_id)
        # inf_indices = tgt == -float('inf')
        # next_aa = np.ones_like(next_aa) * (1 - epsilon ** (1 / (self.cfg.data.peptide_max_len-seq_len))) / (
        #         len(next_aa) - 1)
        # # print('next_aa after', next_aa)
        # next_aa[max_id] = epsilon ** (1 / (self.cfg.data.peptide_max_len-seq_len))
        # # print('next_aa max_id', next_aa)
        # next_aa[tgt[-1, :] == float('-inf')] = 0
        # next_aa /= np.sum(next_aa)
        # # print('next_aa', next_aa)
        # next_aa = np.random.choice(len(next_aa), p=next_aa)
        # print('next_aa', next_aa)

        return max_id

    def backforth_sample(self, pep_status_list, final_node_mass, final_mem, final_rel_mask, device, pos_index, k=4):
        assert k > 0
        batch_size = len(pep_status_list)
        k = pos_index // 2 - k
        # Do Backward k steps: using the past tgt generated by fwd policy
        tgt, glycan_mass, glycan_crossattn_mass, index_list = self.back_sample_input_gen(pep_status_list, device, k, pos_index)
        # initialize with the generated sequence
        pos_index -= k * 2
        true_idx = math.ceil(pos_index / 2)
        # print('pos_index', pos_index, tgt.shape)
        range_tensor = torch.arange(true_idx + 1)
        # print('first sample back tgt', tgt, pos_index)
        with torch.no_grad():
            if dist.is_initialized():
                tgt, past_tgts, past_mems = self.back_model.module.inference_initialize(tgt=tgt,
                                                                                        pos_index=range_tensor.repeat_interleave(
                                                                                            2)[
                                                                                                  1:pos_index + 1].cuda().unsqueeze(
                                                                                            0),
                                                                                        node_mass=final_node_mass[index_list],
                                                                                        rel_mask=final_rel_mask[index_list],
                                                                                        glycan_mass=glycan_mass.squeeze(
                                                                                            0),
                                                                                        mem=final_mem[index_list],
                                                                                        glycan_crossattn_mass=glycan_crossattn_mass.squeeze(
                                                                                            0))
            else:
                tgt, past_tgts, past_mems = self.back_model.inference_initialize(tgt=tgt,
                                                                                 pos_index=range_tensor.repeat_interleave(
                                                                                     2)[
                                                                                           1:pos_index + 1].cuda().unsqueeze(
                                                                                     0),
                                                                                 node_mass=final_node_mass[index_list],
                                                                                 rel_mask=final_rel_mask[index_list],
                                                                                 glycan_mass=glycan_mass.squeeze(0),
                                                                                 mem=final_mem[index_list],
                                                                                 glycan_crossattn_mass=glycan_crossattn_mass.squeeze(
                                                                                     0))

        pep_status_list = self.back_aa_choice(tgt, pep_status_list, )
        for step in range(2, (k+1) * 2, 2):  # assume k-1 is longer than trajectory length
            # print('pos index back', pos_index + step)
            tgt, glycan_mass, glycan_crossattn_mass, mask = self.decoder_inference_input_gen(pep_status_list, device,
                                                                                       usage='back',
                                                                                       max_len=pos_index + step)

            with torch.no_grad():
                if dist.is_initialized():
                    tgt, past_tgts = self.back_model.module.inference_step(tgt=tgt,
                                                                           pos_index=torch.tensor([pos_index + step-1],
                                                                                                  dtype=torch.long,
                                                                                                  device=device),
                                                                           glycan_crossattn_mass=glycan_crossattn_mass,
                                                                           glycan_mass=glycan_mass,
                                                                           past_tgts=past_tgts,
                                                                           past_mems=past_mems,
                                                                           rel_mask=final_rel_mask,
                                                                           is_idx=(pos_index + step) % 2 != 0,
                                                                           )
                else:
                    tgt, past_tgts = self.back_model.inference_step(tgt=tgt,
                                                                    pos_index=torch.tensor([pos_index + step-1],
                                                                                           dtype=torch.long,
                                                                                           device=device),
                                                                    glycan_crossattn_mass=glycan_crossattn_mass,
                                                                    glycan_mass=glycan_mass,
                                                                    past_tgts=past_tgts,
                                                                    past_mems=past_mems,
                                                                    rel_mask=final_rel_mask,
                                                                    is_idx=(pos_index + step) % 2 != 0,
                                                                   )

            pep_status_list = self.back_aa_choice(tgt, pep_status_list)

        # Do Forward k steps
        tgt, glycan_mass, glycan_crossattn_mass, mask, index_list = self.fwd_sample_input_gen(pep_status_list, device, k, pos_index)
        # initialize with the generated sequence
        true_idx = math.ceil(pos_index / 2)
        # print('pos_index', pos_index, tgt.shape)
        range_tensor = torch.arange(true_idx + 1)
        # print('first sample forward tgt', tgt)
        with torch.no_grad():
            if dist.is_initialized():
                tgt, past_tgts, past_mems = self.fwd_model.module.inference_initialize_idx(tgt=tgt,
                                                                                           pos_index=pos_index,
                                                                                           node_mass=final_node_mass[index_list],
                                                                                           rel_mask=final_rel_mask[index_list],
                                                                                           glycan_mass=glycan_mass.squeeze(0),
                                                                                           mem=final_mem[index_list],
                                                                                           glycan_crossattn_mass=glycan_crossattn_mass.squeeze(0),
                                                                                           tgt_mask=mask)
            else:
                tgt, past_tgts, past_mems = self.fwd_model.inference_initialize_idx(tgt=tgt,
                                                                                    pos_index=pos_index,
                                                                                    node_mass=final_node_mass[index_list],
                                                                                    rel_mask=final_rel_mask[index_list],
                                                                                    glycan_mass=glycan_mass.squeeze(0),
                                                                                    mem=final_mem[index_list],
                                                                                    glycan_crossattn_mass=glycan_crossattn_mass.squeeze(0),
                                                                                    tgt_mask=mask)

        pep_status_list = self.next_aa_choice(tgt, pep_status_list, local_search=True)

        for step in range(2, (k+1) * 2, 2):  # assume k-1 is longer than trajectory length
            # print('pos index back', pos_index + step)
            tgt, glycan_mass, glycan_crossattn_mass, mask = self.decoder_inference_input_gen(pep_status_list, device,
                                                                                       usage='fwd', max_len=pos_index+step)

            with torch.no_grad():
                if dist.is_initialized():
                    tgt, past_tgts = self.fwd_model.module.inference_step(tgt=tgt,
                                                                          pos_index=torch.tensor([pos_index + step-1],
                                                                                                 dtype=torch.long,
                                                                                                 device=device),
                                                                          glycan_crossattn_mass=glycan_crossattn_mass,
                                                                          glycan_mass=glycan_mass,
                                                                          past_tgts=past_tgts,
                                                                          past_mems=past_mems,
                                                                          rel_mask=final_rel_mask,
                                                                          is_idx=(pos_index + step) % 2 == 0,
                                                                          tgt_mask=mask)
                else:
                    tgt, past_tgts = self.fwd_model.inference_step(tgt=tgt,
                                                                   pos_index=torch.tensor([pos_index + step-1],
                                                                                          dtype=torch.long,
                                                                                          device=device),
                                                                   glycan_crossattn_mass=glycan_crossattn_mass,
                                                                   glycan_mass=glycan_mass,
                                                                   past_tgts=past_tgts,
                                                                   past_mems=past_mems,
                                                                   rel_mask=final_rel_mask,
                                                                   is_idx=(pos_index + step) % 2 == 0,
                                                                   tgt_mask=mask)
            # print('tgt', tgt)
            pep_status_list = self.next_aa_choice(tgt, pep_status_list, local_search=True)
        return pep_status_list

    def exploration_initializing(self):
        # data entry for each spectra
        encoder_input, decoder_input, seq, precursor_mass, pep_mass, _, _, label, label_mask = next(self.inference_dl)
        # print(len(seq))
        # 由于需要自由探索，对decoder input里面每个input只取<bos>
        if dist.is_initialized():
            mem = self.fwd_model.module.mem_get(**encoder_input)
        else:
            mem = self.fwd_model.mem_get(**encoder_input)
        print('gt', seq)
        gt_seq = torch.zeros((len(self.tokenize_aa_dict) + 1, 1))
        gt_seq[0] = 1
        pep_status_list = [Pep_Inference_Status(idx=i,
                                                label_seq=convert2glycoCT(seq[i]),
                                                pep_mass=pep_mass[i],
                                                precursor_mass=precursor_mass[i],
                                                inference_seq=decoder_input['tgt'][i],
                                                correct_lst=torch.tensor([1]),
                                                gt_seq=gt_seq.T,
                                                back_inf_seq=copy.deepcopy(decoder_input['tgt'][i]),
                                                fwd_inf_seq=copy.deepcopy(decoder_input['tgt'][i])
                                                ) for i in range(len(seq))]
        with torch.no_grad():
            if dist.is_initialized():
                tgt, past_tgts, past_mems = self.fwd_model.module.inference_initialize(mem=mem, **decoder_input)
            else:
                tgt, past_tgts, past_mems = self.fwd_model.inference_initialize(mem=mem, **decoder_input)
        return mem, tgt, past_tgts, past_mems, pep_status_list, encoder_input['node_mass'], encoder_input[
            'rel_mask']  # , decoder_input['product_ion_moverz'], decoder_input['product_ion_mask']#, candidate_label

    def back_aa_choice(self, tgt, pep_status_list):
        tgt = tgt.float().cpu()
        for i, current_status in enumerate(pep_status_list):
            next_aa = self.epsilon_sampling(tgt[i, :, :])
            tgt_current = torch.tensor([next_aa]).cuda()
            current_status.back_inf_seq = torch.cat((current_status.back_inf_seq, tgt_current))
            if len(current_status.back_inf_seq) != len(current_status.inference_seq):
                next_parent_idx = current_status.inference_seq[len(current_status.back_inf_seq) + 1].unsqueeze(-1)
                current_status.back_inf_seq = torch.cat((current_status.back_inf_seq, next_parent_idx))
            # current_status.inference_seq[-1] = tgt_current
            # print('back_inf_seq', current_status.back_inf_seq)
            # print('back sampled', current_status.inference_seq)

            current_status.current_aa_index -= 1
            # features for each time step should be based on prediction
            mass_list = torch.zeros((len(current_status.back_inf_seq)))
            mass_list[2::2] = torch.tensor(
                [self.detokenize_aa_dict[i.item()] for i in current_status.back_inf_seq[2::2]])
            mass_list[1:-1:2] = mass_list[2::2]
            # mass_list_parent = mass_list#torch.cumsum(mass_list, dim=0)
            mass_list_parent = torch.zeros((len(current_status.back_inf_seq)))
            for idx, mass in enumerate(mass_list):
                if idx % 2 == 0 and idx > 0:  # mono
                    mass_list_parent[idx] = mass + mass_list_parent[current_status.back_inf_seq[idx - 1] + 1]
                elif 1 < idx < len(current_status.back_inf_seq):  # parent idx
                    mass_list_parent[idx] = mass_list_parent[current_status.back_inf_seq[idx] + 1]
            current_status.current_mass = [torch.sum(mass_list), mass_list_parent[-1]]
        return pep_status_list

    def next_aa_choice(self, tgt, pep_status_list, finish_pep_status_list=None, local_search=False):
        tgt = tgt.float().cpu()
        # print('next aa tgt', tgt.shape)
        # print('tgt_idx', tgt_idx.shape)
        keep_index = torch.ones(len(pep_status_list), dtype=bool)
        for i, current_status in enumerate(pep_status_list):
            # print('tgt.shape', tgt.shape)
            if tgt.shape[-1] == 1:
                next_aa = tgt[i, -1, :].argmax()
                # print('next_aa before', tgt, next_aa,)
            else:
                next_aa = self.epsilon_sampling(tgt[i, :, :])
            # print('tgt[i, :, :]', tgt[i, :, :])
            tgt_current = torch.tensor([next_aa]).cuda()

            if local_search:
                current_status.fwd_inf_seq = torch.cat((current_status.fwd_inf_seq, tgt_current))
                # print('current_status.fwd_inf_seq', current_status.fwd_inf_seq)
                if len(current_status.back_inf_seq) != len(current_status.fwd_inf_seq):
                    next_parent_idx = current_status.back_inf_seq[len(current_status.fwd_inf_seq) + 1].unsqueeze(-1)
                    current_status.fwd_inf_seq = torch.cat((current_status.fwd_inf_seq, next_parent_idx))
                inf_seq = current_status.fwd_inf_seq
            else:
                current_status.inference_seq = torch.cat((current_status.inference_seq, tgt_current))
                inf_seq = current_status.inference_seq
                # print('inf_seq', inf_seq)
                current_status.correct_lst, current_status.gt_seq = self.g.pre2glycan_labels_inversed_random(
                    inf_seq,
                    copy.deepcopy(current_status.label_seq))
                # print('current_status.gt_seq', current_status.gt_seq)
            current_status.current_aa_index += 1
            # print('next_aa_choice', inf_seq)
            # print('correct_seq', correct_seq)
            # features for each time step should be based on prediction
            mass_list = torch.zeros((len(inf_seq)))
            mass_list[1::2] = torch.tensor(
                [self.detokenize_aa_dict[i.item()] for i in inf_seq[1::2]])
            mass_list_parent = torch.cumsum(mass_list, dim=0)
            for idx, mass in enumerate(mass_list):
                if idx % 2 != 0 and idx > 0:  # mono
                    mass_list_parent[idx] = mass
                elif idx > 1:  # parent idx
                    mass_list_parent[idx] = mass_list_parent[inf_seq[idx] + 1] + \
                                            self.detokenize_aa_dict[inf_seq[idx - 1].item()]
            current_status.current_mass = [torch.sum(mass_list), mass_list_parent[-1]]
            # print('current_status.current_mass',torch.sum(mass_list), current_status.precursor_mass)

            if not local_search:
                if len(inf_seq) % 2 != 0:
                    correct_seq = torch.argmax(current_status.gt_seq, dim=-1)  # [:-1]
                    mass_list = torch.zeros((len(correct_seq)))
                    mass_list[1::2] = torch.tensor(
                        [self.detokenize_aa_dict[i.item()] for i in correct_seq[1::2]])
                    if (abs(torch.sum(mass_list) - current_status.precursor_mass) < 0.01
                        or torch.sum(mass_list) > current_status.precursor_mass) or \
                            len(inf_seq) >= self.cfg.data.peptide_max_len:
                        # (len(inf_seq) > 4 and len(inf_seq) >= 2 * true_pre_idx) or \
                        # print('len(inf_seq', len(inf_seq))
                        keep_index[i] = False
            new_pep_status_list = []
        if not local_search:
            for keep_flag, current_status in zip(keep_index, pep_status_list):
                if keep_flag:
                    new_pep_status_list.append(current_status)
                else:
                    finish_pep_status_list.append(current_status)
            return new_pep_status_list, finish_pep_status_list, keep_index
        else:
            return pep_status_list

    def decoder_inference_input_gen(self, pep_status_list, device, usage='inf', max_len=None):
        tgt = []
        current_mass = []
        current_cross_mass = []
        mask = []
        index_list = []
        for current_status in pep_status_list:
            if usage == 'inf':
                tgt.append(current_status.inference_seq)
            elif usage == 'back':
                # print('len(current_status.back_inf_seq)', current_status.back_inf_seq)
                tgt.append(F.pad(current_status.back_inf_seq, (0, max_len - len(current_status.back_inf_seq))))
                mask.append(F.pad(torch.ones(len(current_status.back_inf_seq), dtype=bool, device=device),
                                  (0, max_len - len(current_status.back_inf_seq))))
            else:
                # print('input gen current_status.fwd_inf_seq', current_status.fwd_inf_seq)

                tgt.append(F.pad(current_status.fwd_inf_seq, (0, max_len - len(current_status.fwd_inf_seq))))
                mask.append(F.pad(torch.ones(len(current_status.fwd_inf_seq)-1, dtype=bool, device=device),
                                  (0, max_len+1- len(current_status.fwd_inf_seq))))
                # print('mask', mask)
            current_cross_mass.append(current_status.current_mass)
            current_mass.append(
                [current_status.current_mass[0], current_status.precursor_mass - current_status.current_mass[0]])
            index_list.append(current_status.idx)
        # print('decoder_inference_input_gen', index_list)
        # print('tgt', tgt)
        # print('current_mass', current_mass, 'current_cross_mass', current_cross_mass)
        # index_list = torch.tensor(index_list, dtype=torch.long).argsort()
        tgt = torch.stack(tgt)
        glycan_mass = torch.tensor(current_mass, dtype=torch.float64, device=device).unsqueeze(1)
        glycan_crossattn_mass = torch.tensor(current_cross_mass, dtype=torch.float64, device=device).unsqueeze(
            1)
        if mask:
            # print('before sort', mask)
            mask = torch.stack(mask)
            # print('after sort', mask)
        return tgt, glycan_mass, glycan_crossattn_mass, mask

    def obtain_fwd_mass(self, reversed_seq):
        mass_list = torch.zeros((len(reversed_seq)))
        mass_list[1::2] = torch.tensor(
            [self.detokenize_aa_dict[i.item()] for i in reversed_seq[1::2]])
        mass_list_parent = torch.cumsum(mass_list, dim=0)
        for idx, mass in enumerate(mass_list):
            if idx % 2 != 0 and idx > 0:  # mono
                mass_list_parent[idx] = mass
            elif idx > 1:  # parent idx
                mass_list_parent[idx] = mass_list_parent[reversed_seq[idx] + 1] + \
                                        self.detokenize_aa_dict[reversed_seq[idx - 1].item()]
        return mass_list, mass_list_parent

    def obtain_back_mass(self, reversed_seq):
        mass_list = torch.zeros((len(reversed_seq)))
        mass_list[2::2] = torch.tensor(
            [self.detokenize_aa_dict[i.item()] for i in reversed_seq[2::2]])
        mass_list[1:-1:2] = mass_list[2::2]
        # mass_list_parent = mass_list#torch.cumsum(mass_list, dim=0)
        mass_list_parent = torch.zeros((len(reversed_seq)))
        for idx, mass in enumerate(mass_list):
            if idx % 2 == 0 and idx > 0:  # mono
                mass_list_parent[idx] = mass + mass_list_parent[reversed_seq[idx - 1] + 1]
            elif 1 < idx < len(reversed_seq):  # parent idx
                mass_list_parent[idx] = mass_list_parent[reversed_seq[idx] + 1]
        return mass_list, mass_list_parent

    def back_sample_input_gen(self, pep_status_list, device, k, max_len):
        tgt = []
        index_list = []
        pep_mass = []
        pep_crossattn_mass = []
        max_len = max_len - 2 * k
        for current_status in pep_status_list:
            reversed_seq = copy.deepcopy(current_status.inference_seq)
            reversed_seq[1::2] = current_status.inference_seq[2::2]
            reversed_seq[2::2] = current_status.inference_seq[1::2]
            reversed_seq = reversed_seq[:-2 * k - 1]
            reversed_seq[0] = len(self.aa_dict)
            # print('reversed_seq', reversed_seq)
            mass_list, mass_list_parent = self.obtain_back_mass(reversed_seq)
            current_status.back_inf_seq = reversed_seq
            tgt.append(F.pad(reversed_seq, (0, max_len - len(reversed_seq))))

            nmass = torch.cumsum(mass_list, dim=0)
            pmass = torch.tensor(mass_list_parent)  # current_status.precursor_mass - nmass
            cmass = current_status.precursor_mass - nmass
            # print('nmass', nmass, 'cmass', cmass, 'pmass', pmass)
            pep_mass_item = F.pad(torch.stack([nmass, cmass], dim=-1), (0, 0, 0, max_len - len(reversed_seq)))
            pep_crossattn_mass_item = F.pad(torch.stack([nmass, pmass], dim=-1), (0, 0, 0, max_len - len(
                reversed_seq)))  # torch.concat([pep_mass_item/i for i in range(1, self.cfg.model.max_charge+1)],dim=-1)

            pep_mass.append(pep_mass_item)
            pep_crossattn_mass.append(pep_crossattn_mass_item)
            index_list.append(current_status.idx)
        # print('back_sample_input_gen', index_list)
        # print('tgt', tgt)
        # print('current_mass', current_mass, 'current_cross_mass', current_cross_mass)
        # index_list = torch.tensor(index_list,dtype=torch.long).argsort()
        tgt = torch.stack(tgt)  # .unsqueeze(-1)
        pep_mass = torch.stack(pep_mass).cuda()
        pep_crossattn_mass = torch.stack(pep_crossattn_mass).cuda()
        return tgt, pep_mass, pep_crossattn_mass,index_list

    def fwd_sample_input_gen(self, pep_status_list, device, k, max_len):
        tgt = []
        index_list = []
        pep_mass = []
        pep_crossattn_mass = []
        mask = []
        for current_status in pep_status_list:
            reversed_seq = copy.deepcopy(current_status.back_inf_seq)
            # print('reversed_seq[1::2]', reversed_seq[1::2])
            reversed_seq[1::2] = current_status.back_inf_seq[2::2]
            reversed_seq[2::2] = current_status.back_inf_seq[1::2]
            reversed_seq = reversed_seq[:-2 * k - 1]
            reversed_seq[0] = 0
            # print('fwd reversed_seq', reversed_seq.shape, len(reversed_seq))
            mask.append(F.pad(torch.ones(reversed_seq.shape[0]-1, dtype=bool, device=device),
                              (0, max_len+1 - reversed_seq.shape[0])))
            current_status.fwd_inf_seq = reversed_seq
            mass_list, mass_list_parent = self.obtain_fwd_mass(reversed_seq)
            tgt.append(F.pad(reversed_seq, (0, max_len - len(reversed_seq))))
            nmass = torch.cumsum(mass_list, dim=0)
            pmass = torch.tensor(mass_list_parent)  # current_status.precursor_mass - nmass
            cmass = current_status.precursor_mass - nmass
            # print('nmass', nmass, 'cmass', cmass, 'pmass', pmass)
            pep_mass_item = F.pad(torch.stack([nmass, cmass], dim=-1), (0, 0, 0, max_len - len(reversed_seq)))
            pep_crossattn_mass_item = F.pad(torch.stack([nmass, pmass], dim=-1), (0, 0, 0, max_len - len(
                reversed_seq)))  # torch.concat([pep_mass_item/i for i in range(1, self.cfg.model.max_charge+1)],dim=-1)

            pep_mass.append(pep_mass_item)
            pep_crossattn_mass.append(pep_crossattn_mass_item)
            index_list.append(current_status.idx)

        # print('tgt', tgt)
        # print('current_mass', current_mass, 'current_cross_mass', current_cross_mass)
        # index_list = torch.tensor(index_list,dtype=torch.long).argsort()
        tgt = torch.stack(tgt)  # .unsqueeze(-1)
        pep_mass = torch.stack(pep_mass).cuda()
        pep_crossattn_mass = torch.stack(pep_crossattn_mass).cuda()
        mask = torch.stack(mask).cuda()
        return tgt, pep_mass, pep_crossattn_mass, mask, index_list

    def decoder_train_input_gen(self, pep_finish_pool, max_len, device):
        fwd_inputs = {'tgt':[], 'glycan_mass':[], 'glycan_crossattn_mass':[],}
        back_inputs = copy.deepcopy(fwd_inputs)
        index_list = []
        label = []
        label_mask = []
        for current_status in pep_finish_pool:
            fwd_input = self.decoder_train_input_gen_helper(current_status[0], max_len, device)
            back_input = self.decoder_train_input_gen_helper(current_status[1], max_len, device)

            fwd_inputs['tgt'].append(fwd_input[0])
            fwd_inputs['glycan_mass'].append(fwd_input[1])
            fwd_inputs['glycan_crossattn_mass'].append(fwd_input[2])

            index_list.append(current_status[0].idx)
            back_inputs['tgt'].append(back_input[0])
            back_inputs['glycan_mass'].append(back_input[1])
            back_inputs['glycan_crossattn_mass'].append(back_input[2])

            label.append(fwd_input[3])
            label_mask.append(fwd_input[4])
        # print('len(glycan_finish_pool)', len(glycan_finish_pool))
        # print('index_list', index_list)
        index_list = torch.tensor(index_list, dtype=torch.long)
        # print('index_list', index_list)
        fwd_inputs['tgt'] = torch.stack(fwd_inputs['tgt'])[index_list].cuda()
        fwd_inputs['glycan_mass'] = torch.stack(fwd_inputs['glycan_mass'])[index_list].cuda()
        fwd_inputs['glycan_crossattn_mass'] = torch.stack(fwd_inputs['glycan_crossattn_mass'])[index_list].cuda()

        back_inputs['tgt'] = torch.stack(back_inputs['tgt'])[index_list].cuda()
        back_inputs['glycan_mass'] = torch.stack(back_inputs['glycan_mass'])[index_list].cuda()
        back_inputs['glycan_crossattn_mass'] = torch.stack(back_inputs['glycan_crossattn_mass'])[index_list].cuda()
        label = torch.stack(label)[index_list].cuda()
        label_mask = torch.stack(label_mask)[index_list].cuda()
        return fwd_inputs, back_inputs, label, label_mask, index_list

    def decoder_train_input_gen_helper(self, current_status, max_len, device):
        # backward and forward input
        gt = current_status.train_seq #torch.argmax(current_status.gt_seq, dim=-1)
        # print('gt', gt.shape,max_len )
        # gt = torch.cat((torch.tensor([len(self.tokenize_aa_dict)]), gt), dim=-1)
        tgt_item = gt.cuda()  # torch.tensor([self.aa_dict[current_status.train_seq[i]] for i in range(len(current_status.train_seq))],dtype=torch.long,device=device)
        # print('tgt_item',tgt_item)
        tgt_item = F.pad(tgt_item, (0, max_len - len(gt)))
        nmass = torch.tensor(current_status.accu_mass_list)
        pmass = torch.tensor(current_status.parent_mass_list)  # current_status.precursor_mass - nmass
        cmass = current_status.precursor_mass - nmass
        pep_mass_item = F.pad(torch.stack([nmass, cmass], dim=-1), (0, 0, 0, max_len - len(gt)))
        pep_crossattn_mass_item = F.pad(torch.stack([nmass, pmass], dim=-1), (0, 0, 0, max_len - len(
            gt)))  # torch.concat([pep_mass_item/i for i in range(1, self.cfg.model.max_charge+1)],dim=-1)
        label_item = current_status.gt_seq[1:, :]
        # print('label_item', label_item,'tgt_item',tgt_item.shape )
        # label_item = candidate_label[current_status.idx,:current_status.correct_num]
        # print('train label_item', label_item.shape,)
        label_item = F.pad(label_item,
                           (0, max(0, max_len - label_item.size(1)), 0, max(0, max_len - label_item.size(0))))
        # print('train label_item', torch.argmax(label_item, dim=-1), )
        label_mask_item = F.pad(torch.ones(len(current_status.train_seq), dtype=bool, device=device),
                                (0, max_len - len(current_status.train_seq)))

        # print('train label_mask_item', label_mask_item, max_len)
        # only keep those after bos
        # print('label_mask_item', label_mask_item)
        # print('pep_crossattn_mass_item', pep_crossattn_mass_item.shape)

        return tgt_item, pep_mass_item, pep_crossattn_mass_item, label_item, label_mask_item

