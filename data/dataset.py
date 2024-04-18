import os
import gzip
import torch
import pickle
from torch.utils.data import Dataset
from gnova.utils.BasicClass import Residual_seq, Ion


class GenovaDataset(Dataset):
    def __init__(self, cfg, aa_dict, spec_header, dataset_dir_path):
        super().__init__()
        self.cfg = cfg
        self.aa_dict = aa_dict
        self.spec_header = spec_header
        self.dataset_dir_path = dataset_dir_path

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        spec_head = dict(self.spec_header.loc[idx])
        with open(os.path.join(self.dataset_dir_path, spec_head['MSGP File Name']), 'rb') as f:
            f.seek(spec_head['MSGP Datablock Pointer'])
            spec = pickle.loads(gzip.decompress(f.read(spec_head['MSGP Datablock Length'])))
        # print(spec.keys())
        node_mass = torch.Tensor(spec['node_mass'])
        precursor_mass = spec_head['Glycan mass']
        pep_mass = spec_head['Pep mass']
        charge = spec_head['Charge']
        # print('node_mass', node_mass.shape)

        node_feature = spec['node_input']['node_feat'][:, 0, :-1]
        node_sourceion = spec['node_input']['node_sourceion'][:, 0]
        node_sourceion[node_sourceion == 10] = 4
        node_sourceion[node_sourceion == 13] = 5

        dist = spec['rel_input']['dist'].long()
        predecessors = spec['rel_input']['predecessors'].long()
        predecessors[predecessors == -9999] = 0

        # decoder input
        graph_label = torch.tensor(spec['graph_label'])
        # if self.cfg.train.task == 'optimal_path':
        #     tgt = [torch.zeros_like(graph_label)]
        # else:
        tgt = [0]  # [self.aa_dict['<bos>']]  # decoder_input['tgt'][:,[0]]
        # decoder_input['pos_index'] = decoder_input['pos_index'][:, [0]]
        glycan_mass_embeded = torch.DoubleTensor([0, precursor_mass])
        glycan_crossattn_mass = torch.concat(
            [torch.DoubleTensor([0, 0]) / i for i in range(1, self.cfg.model.max_charge + 1)], dim=-1)
        # glycan_mass_embeded = torch.DoubleTensor([0])
        # glycan_crossattn_mass = torch.concat(
        #     [glycan_mass_embeded / i for i in range(1, self.cfg.model.max_charge + 1)], dim=-1)

        seq = spec_head['Glycan Stru']
        # pep_mass_embeded = self.peptide_mass_embedding(seq, precursor_mass)
        # pep_crossattn_mass = self.pepmass_crossattn_embeded(seq, precursor_mass)

        return {'node_num': spec_head['Node Number'],
                'node_feature': node_feature,
                'node_sourceion': node_sourceion,
                'node_mass': node_mass,
                'dist': dist,
                'predecessors': predecessors,
                'graph_label': graph_label,
                'tgt': tgt,
                'glycan_mass_embeded': glycan_mass_embeded,
                'glycan_crossattn_mass': glycan_crossattn_mass,
                'seq': seq,
                'precursor_mass': precursor_mass,
                'pep_mass': pep_mass,
                'precursor_charge': charge,
                'psm_index': spec_head['Spec Index']
                }

    def __len__(self):
        return len(self.spec_header)