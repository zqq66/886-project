import os
import torch
import gnova
import math
from torch import nn
from torch import optim
from gnova.optimizer import Lion
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from glypy import Monosaccharide, monosaccharides, Glycan


class Task:
    def __init__(self, cfg, serialized_model_path):
        self.cfg = cfg
        self.serialized_model_path = serialized_model_path
        
        self.i = 0
        
        try:
            dist.init_process_group(backend='nccl')
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
            self.distributed = True
        except ValueError:
            self.local_rank = 0
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
            self.distributed = False

    def initialize(self, *, train_spec_header,train_dataset_dir,val_spec_header,val_dataset_dir):
        self.model = gnova.models.GNova(self.cfg).to(self.device)
        
        self.train_loss_fn = nn.BCEWithLogitsLoss()
        self.eval_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
        
        if self.distributed: self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.train.lr)
        #self.optimizer = Lion(self.model.parameters(), lr=self.cfg.train.lr)
        self.scaler = GradScaler()
        self.persistent_file_name = os.path.join(self.serialized_model_path,self.cfg.wandb.project+'_'+self.cfg.wandb.name+'.pt')
        if os.path.exists(self.persistent_file_name):
            checkpoint = torch.load(self.persistent_file_name,map_location={f'cuda:{0}': f'cuda:{self.local_rank}'})
            if self.distributed: self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else: self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f'load model from {self.persistent_file_name}')
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else: print('no model detected, start from scratch')
        train_spec_header = train_spec_header.sample(frac=1).reset_index(drop=True)
        ds = gnova.data.GenovaDataset(spec_header=train_spec_header, dataset_dir_path=train_dataset_dir)
        # train_idx, val_idx = train_test_split(list(range(len(ds))), test_size=0.1)
        train_idx = list(range(len(ds)))[:math.ceil(len(ds) * 0.9)]  #
        val_idx = list(range(len(ds)))[math.ceil(len(ds) * 0.9):]
        train_set = Subset(ds, train_idx)
        val_set = Subset(ds, val_idx)
        sampler = gnova.data.GNovaBucketBatchSampler(self.cfg, train_spec_header.iloc[train_idx, :])
        train_dl = DataLoader(train_set, batch_sampler=sampler, collate_fn=gnova.data.GenovaCollator(self.cfg),
                              pin_memory=True, num_workers=5)
        self.train_dl = gnova.data.DataPrefetcher(train_dl, self.device)
        # val_sampler = gnova.data.GNovaBucketBatchSampler(self.cfg, train_spec_header.iloc[val_idx, :])

        eval_dl = DataLoader(val_set, batch_size=1, collate_fn=gnova.data.GenovaCollator(self.cfg),
                             pin_memory=True, num_workers=5)
        self.eval_dl = gnova.data.DataPrefetcher(eval_dl, self.device)

    def train_loader(self,train_spec_header,train_dataset_dir):
        ds = gnova.data.GenovaDataset(spec_header=train_spec_header,dataset_dir_path=train_dataset_dir)
        sampler = gnova.data.GNovaBucketBatchSampler(self.cfg,train_spec_header)
        train_dl = DataLoader(ds,batch_sampler=sampler,collate_fn=gnova.data.GenovaCollator(self.cfg),pin_memory=True,num_workers=2)
        train_dl = gnova.data.DataPrefetcher(train_dl,self.device)
        return train_dl

    def eval_loader(self,val_spec_header,val_dataset_dir):
        ds = gnova.data.GenovaDataset(spec_header=val_spec_header,dataset_dir_path=val_dataset_dir)
        if self.distributed:
            sampler = DistributedSampler(ds,shuffle=False)
            eval_dl = DataLoader(ds,batch_size=4,sampler=sampler,collate_fn=gnova.data.GenovaCollator(self.cfg),pin_memory=True,num_workers=2)
        else:
            eval_dl = DataLoader(ds,batch_size=1,collate_fn=gnova.data.GenovaCollator(self.cfg),pin_memory=True,num_workers=2)
        eval_dl = gnova.data.DataPrefetcher(eval_dl,self.device)
        return eval_dl
    
    def model_save(self):
        if self.distributed:
            torch.save({'model_state_dict':self.model.module.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict()},self.persistent_file_name)
        else:
            torch.save({'model_state_dict':self.model.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict()},self.persistent_file_name)

    def train(self):
        total_step = 0
        loss_cum = 0
        total_seq_len = torch.Tensor([0]).to(self.device)
        total_match = torch.Tensor([0]).to(self.device)
        true_positive = torch.Tensor([0]).to(self.device)
        total_positive = torch.Tensor([0]).to(self.device)
        total_true = torch.Tensor([0]).to(self.device)
        for epoch in range(0, self.cfg.train.total_epoch):
            for encoder_input, label, label_mask in self.train_dl:
                total_step += 1
                self.optimizer.zero_grad(set_to_none=True)
                if total_step%self.cfg.train.detect_period == 1: loss_cum = 0
                with autocast():
                    output = self.model(encoder_input=encoder_input).squeeze(-1)
                    loss = self.train_loss_fn(output[label_mask],label[label_mask])
                output = output[label_mask].squeeze(-1)
                label = label[label_mask]
                output = (output > 0.5).float()
                loss_cum += loss
                total_seq_len += label_mask.sum()
                total_match += (output == label).sum()
                true_positive += ((output == label)[label == 1]).sum()
                total_positive += (label == 1).sum()
                total_true += (output == 1).sum()
                loss_cum += loss.item()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # if total_step%self.cfg.train.detect_period == 0: yield loss_cum/self.cfg.train.detect_period, total_step, epoch

                yield loss_cum / total_step, total_step, epoch, \
                    (total_match / total_seq_len).item(), \
                    (true_positive / total_positive).item(), \
                    (true_positive / total_true).item()
                
    def eval(self) -> float:
        loss_cum = torch.Tensor([0]).to(self.device)
        total_seq_len = torch.Tensor([0]).to(self.device)
        total_match = torch.Tensor([0]).to(self.device)
        true_positive = torch.Tensor([0]).to(self.device)
        total_positive = torch.Tensor([0]).to(self.device)
        total_true = torch.Tensor([0]).to(self.device)

        for encoder_input, label, label_mask in self.eval_dl:
            with torch.no_grad():
                output = self.model(encoder_input=encoder_input)
                output = output[label_mask].squeeze(-1)
                label = label[label_mask]
                loss = self.eval_loss_fn(output,label)
                output = (output>0.5).float()
                loss_cum += loss
                total_seq_len += label_mask.sum()
                total_match += (output == label).sum()
                true_positive += ((output == label)[label == 1]).sum()
                total_positive += (label == 1).sum()
                total_true += (output == 1).sum()
        if self.distributed:
            dist.all_reduce(loss_cum)
            dist.all_reduce(total_seq_len)
            dist.all_reduce(total_match)
            dist.all_reduce(true_positive)
            dist.all_reduce(total_positive)
            dist.all_reduce(total_true)
        #wandb.log({'recall eval':(true_positive/total_positive).item(), 'precision eval': (true_positive/total_true).item(), 'accu eval': (total_match/total_seq_len).item()}, step=total_step)
        return (loss_cum/total_seq_len).item(), \
                (total_match/total_seq_len).item(), \
                (true_positive/total_positive).item(), \
                (true_positive/total_true).item()
