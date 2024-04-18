import torch
import torch.nn.functional as F

class RnovaCollator(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        max_peak_number = max([record['product_ion_moverz'].size(0) for record in batch])
        product_ion_mask = torch.stack([torch.BoolTensor([False]*record['product_ion_moverz'].shape[0]+[True]*(max_peak_number-record['product_ion_moverz'].size(0))) for record in batch]).unsqueeze(1).unsqueeze(1)
        product_ion_moverz = torch.stack([F.pad(record['product_ion_moverz'],[0,max_peak_number-record['product_ion_moverz'].size(0)]) for record in batch])
        product_ion_intensity = torch.stack([F.pad(record['product_ion_intensity'],[0,max_peak_number-record['product_ion_intensity'].size(0)]) for record in batch])
        encoder_pos_index = torch.stack([F.pad(record['encoder_pos_index'],[0,max_peak_number-record['encoder_pos_index'].size(0)]) for record in batch])
        charge = torch.LongTensor([record['charge'] for record in batch])
        charge_threshold = [record['charge'] for record in batch]
        
        seq = [record['seq'] for record in batch]
        precursor_mass = [record['precursor_mass'] for record in batch]
        optimal_path = [record['optimal_path'] for record in batch]
        psm_index = [record['psm_index'] for record in batch]
        
        return ({'charge':charge,
                 'product_ion_moverz':product_ion_moverz,
                 'product_ion_intensity':product_ion_intensity,
                 'encoder_pos_index':encoder_pos_index,
                 'product_ion_mask':product_ion_mask,},
                 seq, precursor_mass, optimal_path, charge_threshold, psm_index)
