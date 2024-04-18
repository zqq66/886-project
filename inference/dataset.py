import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.BasicClass import Residual_seq, Ion

class RnovaDataset(Dataset):
    def __init__(self, aa_dict, data, psm_head, tokenize_aa_dict, cfg):
        super().__init__()
        self.aa_dict = aa_dict
        self.data = data
        self.psm_head = psm_head
        self.tokenize_aa_dict = tokenize_aa_dict
        self.single_aa_mask_list = ['N','Q','K','m','F','R','W']
        self.cfg = cfg
        
    def __getitem__(self, idx):
        seq,moverz,charge,experiment,file_id,scan,_,pred_seq,label_seq = self.psm_head.iloc[idx]
        seq = seq[::-1]
        pred_seq = pred_seq.split(' ')[::-1]
        label_seq = label_seq.split(' ')[::-1]
        label_seq = [Residual_seq(block).mass if len(block)>1 or block in self.single_aa_mask_list else block for block in label_seq]
        
        #encodeer input
        product_ion_moverz, product_ion_intensity = self.data[f'{experiment}:{file_id}:{scan}']
        precursor_mass = Ion.precursorion2mass(precursor_ion_moverz=moverz, precursor_ion_charge=charge)
        encoder_pos_index = torch.concat([torch.zeros(len(product_ion_moverz),dtype=torch.long),
                                          torch.arange(1,self.cfg.model.max_charge+2,dtype=torch.long)])
        product_ion_moverz = torch.concat([torch.from_numpy(product_ion_moverz),
                                           torch.tensor([Ion.precursorion2ionmz(moverz,charge,i) if i!=0 else 0 for i in range(self.cfg.model.max_charge+1)])])
        product_ion_intensity = torch.concat([torch.from_numpy(np.log(product_ion_intensity/product_ion_intensity.max())),
                                              torch.tensor([0]*(self.cfg.model.max_charge+1))])
        
        return {'product_ion_moverz':product_ion_moverz,
                'product_ion_intensity':product_ion_intensity,
                'encoder_pos_index': encoder_pos_index,
                'charge':charge,
                'seq':seq, 
                'precursor_mass':precursor_mass,
                'optimal_path':pred_seq,
                'psm_index':f'{experiment}:{file_id}:{scan}'}
        
    def __len__(self):
        return len(self.psm_head)
