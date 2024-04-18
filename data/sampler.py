import random
import numpy as np
import torch.distributed as dist
from torch.utils.data import Sampler
from copy import deepcopy

class RnovaBucketBatchSampler(Sampler):
    """Wraps another sampler to yield a mini-batch of indices.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, cfg, spec_header,
                 shuffle=True, drop_last=True) -> Sampler:
        super().__init__(data_source=None)
        random.seed(0)
        self.cfg = cfg
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.spec_header = spec_header
        self.random_gen = np.random.default_rng(seed=0)
        self.bins_ori = [np.array(self.spec_header[np.logical_and(self.spec_header['Node Number']>self.cfg.sample.bin_boarders[i], \
                                  self.spec_header['Node Number']<=self.cfg.sample.bin_boarders[i+1])].index) \
                                  for i in range(len(self.cfg.sample.bin_boarders)-1)]
        self.bin_batch_size = np.array(self.cfg.sample.bin_batch_size)

    def __iter__(self):
        self.generate_bins()
        return self

    def __next__(self):
        if (self.bins_readpointer).sum()>=self.bin_len.sum(): raise StopIteration
        if self.shuffle:
            sample_left = (self.bin_len - self.bins_readpointer) != 0
            available_bin = np.arange(self.bin_len.size)[sample_left]
            bin_index = random.choices([i for i in available_bin])[0]
        else:
            bin_index = random.choices([i for i in range(self.bin_len.size)], \
                                        weights=(self.bin_len-self.bins_readpointer))[0]
        bin = self.bins[bin_index]
        bin_readpointer = self.bins_readpointer[bin_index]
        index = bin[bin_readpointer:bin_readpointer+self.bin_batch_size[bin_index]]
        self.bins_readpointer[bin_index]+=self.bin_batch_size[bin_index]
        if self.bins_readpointer[bin_index]>self.bin_len[bin_index]: self.bins_readpointer[bin_index]=self.bin_len[bin_index]
        return index
        
    def __len__(self):
        return len(self.spec_header)

    def generate_bins(self):
        if self.shuffle: self.bins = [self.random_gen.permutation(bin) for bin in self.bins_ori]
        else: self.bins = deepcopy(self.bins_ori)
        if dist.is_initialized(): self.bins = [bin[dist.get_rank()::dist.get_world_size()] for bin in self.bins]
        if self.drop_last: self.bins = [bin[:int(len(bin)/batch_size)*batch_size] for batch_size, bin in zip(self.bin_batch_size, self.bins)]
        self.bin_len = np.array([len(bin) for bin in self.bins])
        self.bins_readpointer = np.zeros(len(self.cfg.sample.bin_boarders)-1,dtype=int)