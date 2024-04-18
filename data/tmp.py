import torch
import numpy as np
import copy
from gnova.utils.Glycan import Glycan, convert2glycoCT, Monosaccharide

knapsack_mask_mass =np.array([146.0579088, 162.05282342, 203.07937252, 291.09541651, 307.09033113, ])
id2name = {0: 'hex', 1: 'hexNAc', 2: 'neuAc', 3: 'neuGc', 4: 'fuc'}
name2id = {v:k for (k, v) in id2name.items()}
name2mass ={'hex': 162.05282342015, 'hexNAc': 203.07937252127, 'neuAc': 291.09541650890003, 'neuGc': 307.09033112847, 'fuc': 146.05790880057998}
def find_if_list_mass_is_mono(mass_list, reference):
    ms2_left_boundary = reference.searchsorted(mass_list - 0.2, side='left')
    ms2_right_boundary = reference.searchsorted(mass_list + 0.2, side='right')
    return torch.tensor(ms2_right_boundary > ms2_left_boundary)


def obtain_optimal_path():
    mass_tags = []
    parent_mass = []
    mass_blocks = []

    mass_block = torch.tensor( [ 203.0650,  349.1431,  406.1751,  552.2044,  730.2249,  892.3116,
         933.3072, 1038.4080, 1241.4395, 1419.4634, 1565.5955, 1930.6962])

    print('mass_block', mass_block)
    mass_block = np.append(0, mass_block)
    parent_idx = torch.arange(mass_block.shape[0])
    # print('mass_block', mass_block)
    mass_tag = np.zeros_like(mass_block)
    diff_mass = np.diff(mass_block)
    mass_exist = find_if_list_mass_is_mono(diff_mass, knapsack_mask_mass)
    branch_idx = torch.nonzero(~mass_exist)
    # parent_idx[torch.nonzero(mass_exist)] -= 1
    mass_tag[torch.nonzero(mass_exist)] = diff_mass[torch.nonzero(mass_exist)]
    # print('diff_mass', diff_mass, 'parent_idx', parent_idx, 'mass_tag', mass_tag, 'branch_idx', branch_idx)
    j = 2
    while torch.any(mass_exist) or j < min(5, len(mass_block) - 2):
        diff_mass = mass_block[j:] - mass_block[:-j]
        # TODO: doubled knapsack_mask_mass
        # print('mass_exist', mass_block[j:], mass_block[:-j], diff_mass, parent_idx, j)
        mass_exist = find_if_list_mass_is_mono(diff_mass[branch_idx - j + 1], knapsack_mask_mass)
        parent_idx[branch_idx[mass_exist]] = branch_idx[mass_exist] - j + 1
        mass_tag[branch_idx[mass_exist]] = diff_mass[branch_idx[mass_exist] - j + 1]
        # mass_exist_ori = torch.logical_or(mass_exist_ori, mass_exist)
        branch_idx = branch_idx[torch.where(torch.logical_and(~mass_exist, branch_idx>j-1))[0]]
        j += 1
    non_zero_idx = mass_tag != 0
    print('nonzero', mass_tag, mass_block[parent_idx])
    mass_tags.append(mass_tag[non_zero_idx])
    parent_mass.append(mass_block[parent_idx[non_zero_idx]])
    # print('mass_tag', mass_tag[non_zero_idx])
    # print('parent_idx', mass_block[parent_idx[non_zero_idx]], )
    # print('mass_block', mass_block)
    mass_blocks.append(mass_block)
    # print(stop)
    return mass_tags, parent_mass


def generate_sequence():
    mass_tags, parent_masses = obtain_optimal_path()

    seq = '(N(F)(N(H(H(N(H)))(H(N(H(H)))))))'
    root = convert2glycoCT(seq)
    root.set_index(0)
    mass_tag, parent_mass = mass_tags[0], parent_masses[0]
    # mass_tag = np.array([203.08320618,146.02760315,162.03417969,162.05859375,146.03131104])
    # parent_mass = np.array([[  0.        ,203.08320618,568.230896,730.26507568,892.32366943]])
    print(seq, mass_tag, parent_mass)

    unassigned_node = copy.copy(root.children)
    gt = torch.zeros((32))
    mass_list = torch.zeros((32))
    pos_index = 0
    for i, (m, p) in enumerate(zip(mass_tag, parent_mass)):
        m = int(m)
        unassigned_node = [i for i in unassigned_node if i.index is None]
        unassigned_node_mass = [int(name2mass[r.name]) for r in unassigned_node]
        print(m, unassigned_node_mass)

        if m in unassigned_node_mass:
            node_idx = unassigned_node_mass.index(m)
            print(m, unassigned_node_mass, node_idx)
            new_node = unassigned_node.pop(node_idx)
            print(new_node.name)
            new_node.set_index(pos_index * 2 + 1)
            new_node_id = name2id[new_node.name]
            gt[new_node.index] = new_node_id
            mass_list[new_node.index] = name2mass[new_node.name]
            gt[new_node.index + 1] = new_node.parent.index
            unassigned_node += copy.copy(new_node.children) if new_node.children is not None else []
            pos_index += 1
        else:
            if (sum(mass_list) - p) > int(min(name2mass.values())):
                continue
            m_idx = [int(i) for i in name2mass.values()].index(m)
            possible_mono = root.find_first_unassigned_mono_with_mf(id2name[m_idx])
            if possible_mono is None:
                continue
            parent_mono = copy.copy(possible_mono)
            while parent_mono.parent.index is None:
                parent_mono = parent_mono.parent
            parent_monos = copy.copy(parent_mono.parent.children)
            parent_mono = parent_monos.pop(0)
            while parent_mono != possible_mono:
                parent_mono.set_index(pos_index * 2 + 1)
                gt[parent_mono.index] = name2id[parent_mono.name]
                mass_list[parent_mono.index] = name2mass[parent_mono.name]
                gt[parent_mono.index + 1] = parent_mono.parent.index
                parent_monos += copy.copy(parent_mono.children) if parent_mono.children is not None else []
                pos_index += 1
                parent_mono = parent_monos.pop(0)

            possible_mono.set_index(pos_index * 2 + 1)
            gt[possible_mono.index] = name2id[possible_mono.name]
            mass_list[possible_mono.index] = name2mass[possible_mono.name]
            gt[possible_mono.index + 1] = possible_mono.parent.index
            parent_monos += copy.copy(possible_mono.children) if possible_mono.children is not None else []
            pos_index += 1
            unassigned_node += parent_monos
            unassigned_node += copy.copy(possible_mono.children) if possible_mono.children is not None else []

        print('gt', gt)
    unassigned_node = [i for i in unassigned_node if i.index is None]
    while unassigned_node:
        new_node = unassigned_node.pop(0)
        print(new_node.name)
        new_node.set_index(pos_index * 2 + 1)
        new_node_id = name2id[new_node.name]
        gt[new_node.index] = new_node_id
        mass_list[new_node.index] = name2mass[new_node.name]
        gt[new_node.index + 1] = new_node.parent.index
        unassigned_node += copy.copy(new_node.children) if new_node.children is not None else []
        unassigned_node = [i for i in unassigned_node if i.index is None]
        pos_index += 1
        print('gt', gt)


if __name__ == '__main__':
    generate_sequence()