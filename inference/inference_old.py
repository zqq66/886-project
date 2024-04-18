import torch
from dataclasses import dataclass, field
from utils.BasicClass import Residual_seq, Composition
import operator
from copy import deepcopy

@dataclass
class Pep_Inference_Status:
    psm_idx: str
    idx: int
    inference_seq: list[str]
    label_seq: list[str]
    precursor_mass: float
    optimal_path: list
    confirm_mass: float
    ms1_threshold: float
    ms2_threshold: float
    confirm_mass_block: float = 0
    pred_seq_block_index: int = 0
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
    
    def __iter__(self):
        self.inference_dl = iter(self.inference_dl_ori)
        return self

    def __next__(self):
        pep_finish_pool = {}
        tgt, mems_ori, pep_status_list, product_ion_moverz_ori, product_ion_mask_ori, device = self.exploration_initializing()
        pep_status_list, pep_finish_pool = self.next_aa_choice(tgt, pep_status_list, pep_finish_pool)
        if len(pep_status_list)<=0: return pep_finish_pool
        tgt, pep_mass, pep_crossattn_mass = self.decoder_inference_input_gen(pep_status_list, device)
        mem, product_ion_moverz, product_ion_mask = self.past_input_stacker(pep_status_list, mems_ori, product_ion_moverz_ori, product_ion_mask_ori)
        while len(pep_status_list)>0:
            with torch.inference_mode():
                tgt = self.model.tgt_get(tgt=tgt,
                                         src=mem,
                                         pos_index=torch.arange(tgt.shape[-1],dtype=torch.long,device=device).unsqueeze(0),
                                         pep_crossattn_mass=pep_crossattn_mass,
                                         pep_mass=pep_mass,
                                         product_ion_mask=product_ion_mask,
                                         product_ion_moverz=product_ion_moverz)
            pep_status_list, pep_finish_pool = self.next_aa_choice(tgt, pep_status_list, pep_finish_pool)
            if len(pep_status_list)<=0: break
            tgt, pep_mass, pep_crossattn_mass = self.decoder_inference_input_gen(pep_status_list, device)
            mem, product_ion_moverz, product_ion_mask = self.past_input_stacker(pep_status_list, mems_ori, product_ion_moverz_ori, product_ion_mask_ori)
        
        return pep_finish_pool
    
    def exploration_initializing(self):
        encoder_input, seq, precursor_mass, optimal_paths, charge_threshold, psm_index = next(self.inference_dl)
        with torch.inference_mode(): mem = self.model.mem_get(**encoder_input)
        
        pep_status_list = []
        for i in range(len(seq)):
            pool = Pep_Inference_BeamPool(max_size=self.cfg.inference.beam_size)
            pep_status_list.append(pool)
            optimal_path = []
            confirm_mass = 0
            
            if self.cfg.inference.ms1_threshold_unit=='ppm': 
                ms1_threshold = (precursor_mass[i]+Composition('H2O').mass+\
                                 charge_threshold[i]*Composition('proton').mass) * self.cfg.inference.ms1_threshold * 1e-6
            elif self.cfg.inference.ms1_threshold_unit=='Th': ms1_threshold = self.cfg.inference.ms1_threshold
            else: raise NotImplementedError
            
            if self.cfg.inference.ms2_threshold_unit=='Th': ms2_threshold = self.cfg.inference.ms2_threshold*2
            else: raise NotImplementedError

            for block in optimal_paths[i]:
                try:
                    optimal_path.append(float(block))
                except ValueError:
                    optimal_path.append(block)
                    confirm_mass+=Residual_seq(block).mass
            pool.put(Pep_Inference_Status(psm_idx=psm_index[i], idx=i, label_seq=seq[i], 
                                          precursor_mass=precursor_mass[i],
                                          ms1_threshold = ms1_threshold,
                                          ms2_threshold = ms2_threshold,
                                          optimal_path = optimal_path,
                                          confirm_mass = confirm_mass,
                                          inference_seq=['<bos>']))
        tgt, pep_mass, pep_crossattn_mass = self.decoder_inference_input_gen(pep_status_list, mem.device)
        
        with torch.inference_mode():
            tgt = self.model.tgt_get(tgt=tgt,
                                    src=mem, 
                                    pos_index=torch.tensor([0],dtype=torch.long,device=mem.device), 
                                    product_ion_moverz=encoder_input['product_ion_moverz'], 
                                    product_ion_mask=encoder_input['product_ion_mask'],
                                    pep_mass=pep_mass,  
                                    pep_crossattn_mass=pep_crossattn_mass)
        
        return tgt, mem, pep_status_list, encoder_input['product_ion_moverz'].clone(), encoder_input['product_ion_mask'].clone(), mem.device
        
    def next_aa_choice(self, tgt, pep_status_list, pep_finish_pool):
        i = 0
        new_pep_status_list = []
        tgt = tgt.float().cpu().numpy()
        for current_status_pool in pep_status_list:
            pool = Pep_Inference_BeamPool(max_size=current_status_pool.max_size)
            for current_status in current_status_pool.pool:
                next_aa = tgt[i,-1]
                knapsack_mask, block_decimal = self.knapsack_mask_builder(current_status)
                for aa_id, knapsack_mask_flag in zip(self.detokenize_aa_dict, knapsack_mask):
                    if knapsack_mask_flag:
                        current_status_new = deepcopy(current_status)
                        current_status_new.inference_seq += [self.detokenize_aa_dict[aa_id]]
                        current_status_new.current_mass += Residual_seq(self.detokenize_aa_dict[aa_id]).mass
                        current_status_new.total_score += next_aa[aa_id]
                        current_status_new.total_inference_len+=1
                        #current_status_new.score = current_status_new.total_score/current_status_new.total_inference_len
                        current_status_new.score = current_status_new.total_score
                        current_status_new.score_list += [next_aa[aa_id]]
                        
                        if block_decimal:
                            current_status_new.confirm_mass+=Residual_seq(self.detokenize_aa_dict[aa_id]).mass
                            current_status_new.confirm_mass_block+=Residual_seq(self.detokenize_aa_dict[aa_id]).mass
                            if abs(current_status_new.optimal_path[current_status_new.pred_seq_block_index] - \
                                   current_status_new.confirm_mass_block)<10:
                                current_status_new.pred_seq_block_index += 1
                                current_status_new.confirm_mass_block = 0

                        if abs(current_status_new.precursor_mass-current_status_new.current_mass)<5 or \
                            len(current_status_new.inference_seq)>=self.cfg.data.peptide_max_len:
                            if current_status_new.idx in pep_finish_pool:
                                if pep_finish_pool[current_status_new.idx].score<current_status_new.score:
                                    pep_finish_pool[current_status_new.idx] = Pep_Finish_Status(psm_idx=deepcopy(current_status_new.psm_idx),
                                                                                                inference_seq=''.join(deepcopy(current_status_new.inference_seq[1:])),
                                                                                                label_seq=deepcopy(current_status_new.label_seq),
                                                                                                score=current_status_new.score,
                                                                                                score_list=deepcopy(current_status_new.score_list))
                            else:
                                pep_finish_pool[current_status_new.idx] = Pep_Finish_Status(psm_idx=deepcopy(current_status_new.psm_idx),
                                                                                            inference_seq=''.join(deepcopy(current_status_new.inference_seq[1:])),
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
        if type(current_status.optimal_path[current_status.pred_seq_block_index])==str:
            knapsack_mask = set(current_status.optimal_path[current_status.pred_seq_block_index])
            current_status.pred_seq_block_index += 1
            current_status.confirm_mass_block = 0
            block_decimal = False
        else:
            retain_mass = current_status.precursor_mass-current_status.confirm_mass
            retain_mass_block = current_status.optimal_path[current_status.pred_seq_block_index]-current_status.confirm_mass_block
            
            if retain_mass+current_status.ms1_threshold<self.knapsack_mask_mass.max():
                ms1_left_boundary = self.knapsack_mask_mass.searchsorted(retain_mass-current_status.ms1_threshold,side='left')
                ms1_right_boundary = self.knapsack_mask_mass.searchsorted(retain_mass+current_status.ms1_threshold,side='right')
                ms1_knapsack_mask = set(''.join(self.knapsack_mask_aa[ms1_left_boundary:ms1_right_boundary].tolist()))
            else:
                ms1_knapsack_mask = set(self.detokenize_aa_dict.values())

            if retain_mass+current_status.ms2_threshold<self.knapsack_mask_mass.max():
                ms2_left_boundary = self.knapsack_mask_mass.searchsorted(retain_mass_block-current_status.ms2_threshold,side='left')
                ms2_right_boundary = self.knapsack_mask_mass.searchsorted(retain_mass_block+current_status.ms2_threshold,side='right')
                ms2_knapsack_mask = set(''.join(self.knapsack_mask_aa[ms2_left_boundary:ms2_right_boundary].tolist()))
            else:
                ms2_knapsack_mask = set(self.detokenize_aa_dict.values())

            knapsack_mask = set.intersection(ms1_knapsack_mask,ms2_knapsack_mask)
            block_decimal = True
        knapsack_mask = [True if aa in knapsack_mask else False for aa in self.detokenize_aa_dict.values()]
        return knapsack_mask, block_decimal
    
    def decoder_inference_input_gen(self, pep_status_list, device):
        tgt = []
        current_mass = []
        for current_status_pool in pep_status_list:
            for current_status in current_status_pool.pool:
                tgt.append(torch.tensor([self.aa_dict[aa] for aa in current_status.inference_seq],dtype=torch.long, device=device))
                nterm_mass = torch.tensor([Residual_seq(aa).mass if aa!='<bos>' else 0 for aa in current_status.inference_seq], dtype=torch.double, device=device).cumsum(-1)
                cterm_mass = current_status.precursor_mass - nterm_mass
                current_mass.append(torch.stack([nterm_mass,cterm_mass],dim=1))
        tgt = torch.stack(tgt)
        pep_mass = torch.stack(current_mass)
        pep_crossattn_mass = torch.concat([pep_mass/i for i in range(1,self.cfg.model.max_charge+1)],dim=-1)
        return tgt, pep_mass, pep_crossattn_mass

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