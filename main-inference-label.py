import os
import torch
import pickle
from torch import optim
import pandas as pd
import itertools

import numpy as np
from gnova.utils.BasicClass import Composition
from torchvision.ops import sigmoid_focal_loss
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from gnova.models.rnova import Rnova
from gnova.data.dataset import GenovaDataset
from gnova.data.collator import GenovaCollator
from gnova.data.prefetcher import DataPrefetcher
from gnova.inference import Inference_label_path, Inference_label
from gnova.data.sampler import RnovaBucketBatchSampler
from gnova.utils.Glycan import Glycan, convert2glycoCT, Monosaccharide

import hydra
import json
import glypy
import gzip
import wandb
from omegaconf import DictConfig
try:
    ngpus_per_node = torch.cuda.device_count()
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
except ValueError:
    rank = 0
    local_rank = "cuda" if torch.cuda.is_available() else "cpu"

mono_composition = {
    'hex': Composition('C6H12O6') - Composition('H2O'),
    'hexNAc': Composition('C8H15O6N') - Composition('H2O'),
    'neuAc': Composition('C11H19O9N') - Composition('H2O'),
    'neuGc': Composition('C11H19O10N') - Composition('H2O'),
    'fuc': Composition('C6H12O5') - Composition('H2O'),
}
glycoCT_dict = {
    'Man': 0,
    'GlcNAc': 1,
    'NeuAc':2,
    'NeuGc': 3,
    'Fuc': 4,
    'Xyl': 5
}
aa_dict = {aa:i for i, aa in enumerate(mono_composition)}
# aa_dict['<pad>'] = 0
aa_dict['<bos>'] = len(aa_dict)
tokenize_aa_dict = {i:aa for i, aa in enumerate(mono_composition)}
detokenize_aa_dict = {i: aa.mass for i, aa in enumerate(mono_composition.values())}
detokenize_aa_dict[len(detokenize_aa_dict)] = 0
'''
run = wandb.init(
        name='rl-optimal-path-ordered-new-embed-branch8.pt',

        # Set the project where this run will be logged
        project="rl",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": 1e-4,
        })
    '''
def convert2glycoCT_str(structure_encoding):
    idx = 0
    # for s in structure_encoding:
    #     if s.islower():
    #         structure_encoding = structure_encoding.replace(s, ']')
    #     elif s.isupper():
    #         structure_encoding = structure_encoding.replace(s, '[')
    structure_encoding = structure_encoding.replace(')', ']')
    structure_encoding = structure_encoding.replace('(', '[')
    p_lst= ['H','N','A','G','F']
    for i in p_lst:
        if i in structure_encoding:
            structure_encoding = structure_encoding.replace(i,str(p_lst.index(i) + 1))
    for s in structure_encoding[1:]:
        if s == '[':
            temp_lst = list(structure_encoding)
            temp_lst.insert(idx + 1, ',')
            idx += 2
            structure_encoding = "".join(temp_lst)
        else:
            idx += 1
    struc_lst = json.loads(structure_encoding)

    root = list(glycoCT_dict.keys())[struc_lst[0] - 1]
    glycan = glypy.Glycan(root=glypy.monosaccharides[root])
    glycan = construct_glycan(glycan.root, glycan, struc_lst[1:], 0)
    return glycan


def construct_glycan(root, glycan, struc_lst, cur_idx):

    for i, s in enumerate(struc_lst):
        mono = glypy.monosaccharides[list(glycoCT_dict.keys())[s[0]-1]]
        root.add_monosaccharide(mono)
        next_idx = cur_idx+ 1
        glycan.reindex(method='dfs')
        root2 = glycan[next_idx]
        construct_glycan(root2, glycan, s[1:], next_idx)
    return glycan

def evaluate(inference, rank, cfg):
    g = Glycan(glycoCT_dict)
    tar_more_peak = []
    peak_recall = []
    peak_precision = []
    frag_recall = []
    frag_precision = []
    topology_matches = []
    all_ion_matches = []
    correct_structure = []
    all_q_checked = []
    all_q_non_zero = []
    n_saccharide_matches = []
    pre_matching_ratios = []
    tar_matching_ratios = []
    pre_more_peak = []
    tar_more_peak = []
    num_target_y = 0
    tar_more_peak_psm = []
    for i, finish_pool in enumerate(inference,start=0):
        for pool in finish_pool:
            # print('finish_pool', finish_pool)
            (inf_seq, label_seq, psm_idx) = pool
            inf_seq = inf_seq.cpu().squeeze()
            # print(pool.psm_idx)
            print(psm_idx, label_seq, inf_seq)
            glycan_str = g.convert_pre2glycoCT_inversed(inf_seq)
            glycan = glypy.io.glycoct.loads(glycan_str)
            tar_glycan = convert2glycoCT_str(label_seq)
            topology_match = glycan.topological_equality(tar_glycan)
            # n_saccharide_match = glypy.algorithms.similarity.n_saccharide_similarity(glycan,tar_glycan)
            # n_saccharide_matches.append(n_saccharide_match)
            # print('n_saccharide_match', n_saccharide_match)
            topology_matches.append(1 if topology_match else 0)
            # try:
            peak_tar, frag_tar = find_matched_peak(tar_glycan, psm_idx, cfg)
            peak_pre, frag_pre = find_matched_peak(glycan, psm_idx, cfg)
            num_target_y += len(frag_tar)
            tp_peak = peak_pre.intersection(peak_tar)
            # print(peak_tar, frag_tar, peak_pre, frag_pre)
            if len(peak_pre) == 0:
                print(label_seq)
            # except Exception as e:
            #     print(e)
            #
            #     continue
            peak_precision.append(len(tp_peak)/ max(len(peak_pre), 1))
            peak_recall.append(len(tp_peak) /max(len(peak_tar), 1))
            distinct_pre_matched = peak_pre-peak_tar
            distinct_tar_matched = peak_tar-peak_pre
            pre_more_peak.append(1 if len(distinct_pre_matched)>0 else 0)
            tar_more_peak.append(1 if len(distinct_tar_matched)>0 else 0)
            if len(distinct_tar_matched)>0:
                tar_more_peak_psm.append(psm_idx)
            print('distinct_pre_matched', distinct_pre_matched)
            print('distinct_tar_matched', distinct_tar_matched)
            # print('frag_pre', frag_pre)
            # pre_matching_ratio = len(peak_pre)/len(frag_pre)
            # pre_matching_ratios.append(pre_matching_ratio)
            # tar_matching_ratio = len(peak_tar)/len(frag_tar)
            # tar_matching_ratios.append(tar_matching_ratio)
            tp_frag = frag_tar.intersection(frag_pre)
            all_ion_matches.append(1 if frag_pre == frag_tar else 0)
            if frag_pre == frag_tar:
                correct_structure.append(label_seq)
                print('correct')

            frag_precision.append(len(tp_frag) / max(1, len(frag_pre)))
            frag_recall.append(len(tp_frag) / max(1, len(frag_tar)))
            # print('frag_tar', frag_tar, 'frag_pre', frag_pre, frag_pre-frag_tar)
            # if len(peak_precision):
            #     wandb.log({
            #                   'peak_precision': sum(peak_precision)/len(peak_precision),
            #                    'peak_recall':sum(peak_recall)/len(peak_recall),
            #                    'frag_precision':sum(frag_precision)/len(frag_precision),
            #                    'frag_recall': sum(frag_recall)/len(frag_recall),
            #                'all ion match': sum(all_ion_matches) / len(all_ion_matches),
            #                # 'topological match': sum(topology_matches) / len(topology_matches),
            #                # 'prediction rate': sum(inference.prediction_rate)/len(inference.prediction_rate),
            #                # 'n_saccharide_match': sum(n_saccharide_matches)/len(n_saccharide_matches),
            #                # 'tar_matching_ratio': sum(tar_matching_ratios)/len(tar_matching_ratios),
            #                # 'pre_matching_ratio': sum(pre_matching_ratios)/len(pre_matching_ratios),
            #                # 'pre_more_peak': sum(pre_more_peak)/len(pre_more_peak),
            #                # 'tar_more_peak': sum(tar_more_peak)/len(tar_more_peak)
            #                # 'all_q_matched': sum(all_q_checked)/len(all_q_checked),
            #                # 'q_accuracy': sum(all_q_non_zero)/len(all_q_non_zero),
            #                # 'total prediction': len(peak_precision)
            #                })
        # if i == 10:
            if len(peak_precision):
                print('peak_precision', sum(peak_precision)/len(peak_precision), 'peak_recall', sum(peak_recall)/len(peak_recall))
                print('frag_precision', sum(frag_precision)/len(frag_precision), 'frag_recall', sum(frag_recall)/len(frag_recall))
                print('all ion match', sum(all_ion_matches)/len(all_ion_matches), 'topological match', sum(topology_matches)/len(topology_matches))
        # break

    print('correct prediction', correct_structure)
    print('num_target_y', num_target_y)
    return tar_more_peak_psm
def find_matched_peak(glycan, idx, cfg):
    train_spec_header = pd.read_csv(cfg.test_spec_header_path, index_col='Spec Index')
    spec_head = dict(train_spec_header.loc[idx])
    # print("spec_head['MSGP File Name']", spec_head['MSGP File Name'][0])
    with open(os.path.join(cfg.test_dataset_dir, spec_head['MSGP File Name']), 'rb') as f:
        f.seek(spec_head['MSGP Datablock Pointer'])
        spec = pickle.loads(gzip.decompress(f.read(spec_head['MSGP Datablock Length'])))
    mass_error_da = 0.02
    mass_error_ppm = 5
    resolution = 1e3
    mz0_array = torch.tensor(spec['node_mass'])
    # fragments = [ion.mass - Composition('H2O').mass for ion in glycan.fragments(kind='Y')] #,max_cleavages=5
    glycan_b_set = set()
    glycan_y_set = set()
    fragments = []
    for links, frags in itertools.groupby(glycan.fragments(), lambda f: f.link_ids.keys()):
        y_ion, b_ion = list(frags)
        y_mass_reduced = y_ion.mass - Composition('H2O').mass
        b_mass_int = int(round(b_ion.mass * resolution))
        y_mass_int = int(round(y_mass_reduced * resolution))
        glycan_b_set.add(b_mass_int)
        glycan_y_set.add(y_mass_int)
        fragments.append(y_mass_reduced)
    glycopeptide_array = torch.tensor(fragments)
    num_mz0 = len(mz0_array)
    num_glyco = len(fragments)
    # print('fragments',fragments)
    glycopeptide_array = torch.broadcast_to(glycopeptide_array, (num_mz0, num_glyco))
    # mz0_array = torch.tensor(mz0_list)
    mz0_array = torch.broadcast_to(mz0_array, (num_glyco, num_mz0))
    mz0_array = torch.transpose(mz0_array, 0, 1)
    delta = torch.abs(mz0_array - glycopeptide_array)
    # print(delta)
    glycan_mass = spec_head['Glycan mass']

    mass_threshold = mass_error_da + mass_error_ppm * glycan_mass * 1e-6
    peak_set = set(mz0_array[delta<mass_threshold].tolist())
    fragments = [int(round(y * resolution)) for y in fragments]

    return peak_set, glycan_y_set

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg:DictConfig):
    train_spec_header = pd.read_csv(cfg.test_spec_header_path)
    train_ds = GenovaDataset(cfg, aa_dict, spec_header=train_spec_header, dataset_dir_path=cfg.test_dataset_dir)
    collator = GenovaCollator(cfg)
    train_sampler = RnovaBucketBatchSampler(cfg, train_spec_header)
    train_dl = DataLoader(train_ds,batch_sampler=train_sampler,collate_fn=collator,num_workers=4,pin_memory=True)
    train_dl = DataPrefetcher(train_dl,local_rank)

    mass_list = [0]+list(detokenize_aa_dict.values())[:-1]

    model = Rnova(cfg, torch.tensor(mass_list,device=local_rank), detokenize_aa_dict).to(local_rank)
    new_state_dict = {}
    state_dict = torch.load('save/best/rl-optimal-path-ordered-new-embed-branch8.pt',
                            map_location='cuda:0')  # rl-best-pos9.pt
    for key in state_dict.keys():
        new_key = key.replace('module.', '')  # Remove the 'module.' prefix
        new_state_dict[new_key] = state_dict[key]
    model.load_state_dict(new_state_dict)
    print('model loaded')
    if dist.is_initialized(): model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    # optimizer = Lion(params=model.parameters(),lr=cfg.test.lr,weight_decay=cfg.test.weight_decay)
    knapsack_mask = dict()
    knapsack_mask['mass'] = np.array(list(detokenize_aa_dict.values()))[:-1]
    knapsack_mask['aa_composition'] =  np.array(list(tokenize_aa_dict.keys()))
    inference = Inference_label(cfg, model, train_dl, aa_dict, tokenize_aa_dict, detokenize_aa_dict, knapsack_mask)
    loss_fn = sigmoid_focal_loss#FocalLoss(alpha=0.25, ) torch.nn.MSELoss()
    tar_more_peak_psm = evaluate( inference, rank, cfg)
    # mask = train_spec_header['Spec Index'].isin(list(tar_more_peak_psm))
    # canbeconverted = train_spec_header[mask]
    # canbeconverted.to_csv(cfg.test_dataset_dir+'/test_branch8_path.csv', index=False)


if __name__ == '__main__':
    main()
