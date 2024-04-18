import torch
from torch.nn.functional import pad


class GenovaCollator(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        nodenum = [record['node_num'] for record in batch]
        max_nodenum = max(nodenum)

        node_feature = torch.stack(
            [pad(record['node_feature'], (0, 0, 0, max_nodenum - nodenum[i])) for i, record in enumerate(batch)])
        node_sourceion = torch.stack(
            [pad(record['node_sourceion'], (0, max_nodenum - nodenum[i])) for i, record in enumerate(batch)])
        node_mass = torch.stack(
            [pad(record['node_mass'], (0, max_nodenum - nodenum[i])) for i, record in enumerate(batch)])
        node_mass[:, 0] = 0
        dist = torch.stack(
            [pad(record['dist'], (0, max_nodenum - nodenum[i], 0, max_nodenum - nodenum[i])) for i, record in
             enumerate(batch)])
        predecessors = torch.stack(
            [pad(record['predecessors'], (0, max_nodenum - nodenum[i], 0, max_nodenum - nodenum[i])) for i, record in
             enumerate(batch)]).unsqueeze(-1)
        rel_mask = torch.stack(
            [pad(torch.ones_like(record['node_mass'], dtype=bool), (0, max_nodenum - nodenum[i])) for i, record in
             enumerate(batch)]).unsqueeze(1).unsqueeze(-1)
        label = torch.stack(
            [pad(record['graph_label'], (0, max_nodenum - nodenum[i])) for i, record in enumerate(batch)])
        label_mask = torch.stack(
            [pad(torch.ones_like(record['graph_label'], dtype=bool), (0, max_nodenum - nodenum[i])) for i, record in
             enumerate(batch)])

        encoder_input = {
            'node_feature': node_feature,
            'node_sourceion': node_sourceion,
            'node_mass': node_mass,
            'dist': dist,
            'predecessors': predecessors,
            'rel_mask': rel_mask
        }

        # max_tgt_len = max([record['tgt'].size(0) for record in batch])
        tgt = torch.stack([torch.tensor(record['tgt']) for record in batch])
        glycan_mass_embeded = torch.stack([record['glycan_mass_embeded'] for record in batch])
        glycan_crossattn_mass = torch.stack([record['glycan_crossattn_mass'] for record in batch])
        pos_index = torch.arange(0, 1).unsqueeze(0)

        seq = [record['seq'] for record in batch]
        precursor_mass = [record['precursor_mass'] for record in batch]
        pep_mass = [record['pep_mass'] for record in batch]
        charge = [record['precursor_charge'] for record in batch]

        psm_index = [record['psm_index'] for record in batch]
        decoder_input = {'tgt': tgt,
                         'pos_index': pos_index,
                         'glycan_crossattn_mass': glycan_crossattn_mass,
                         'glycan_mass': glycan_mass_embeded,
                         'node_mass': node_mass,
                         'rel_mask': rel_mask}

        return encoder_input, decoder_input, seq, precursor_mass, pep_mass, psm_index, charge, label, label_mask