import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.utils.checkpoint import checkpoint

class DeepNorm(nn.Module):
    def __init__(self, normalized_shape, alpha, dropout_rate) -> None:
        super().__init__()
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout_rate)
        self.ln = nn.LayerNorm(normalized_shape)
    
    def forward(self, x, postx):
        return self.ln(x*self.alpha + self.dropout(postx))

class MultiHeadAttn(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 max_charge: int,
                 n_head_per_mass: int,
                 key_size: int,
                 alpha: float,
                 beta: float,
                 dropout_rate: float):
        """_summary_

        Args:
            hidden_size (int): same with transformer hidden_size
            d_relation (int): relation matrix dimention
            num_head (int): same with transformer num_head
            layer_num (int): How many layers in total
        """
        super().__init__()
        assert hidden_size % 8 == 0
        assert key_size % 8 ==0 and key_size <= 128

        self.key_size = key_size
        self.n_head_projection = (2*max_charge+1)*n_head_per_mass

        # 使用Pre Norm，降低训练难度
        self.linear_q = nn.Linear(hidden_size, self.key_size*self.n_head_projection)
        self.linear_k = nn.Linear(hidden_size, self.key_size*self.n_head_projection)
        self.linear_v = nn.Linear(hidden_size, self.key_size*self.n_head_projection)
        self.output_layer = nn.Linear(self.key_size*self.n_head_projection, hidden_size)
        self.dn = DeepNorm(hidden_size, alpha, dropout_rate)

        nn.init.xavier_normal_(self.linear_v.weight, gain=beta)
        nn.init.xavier_normal_(self.output_layer.weight, gain=beta)

    def forward(self,x,product_ion_info_query,product_ion_info_key,src_key_padding_mask):
        batch_size = x.size(0)
        q = self.linear_q(x).view(batch_size, -1, self.n_head_projection, self.key_size)
        k = self.linear_k(x).view(batch_size, -1, self.n_head_projection, self.key_size)
        v = self.linear_v(x).view(batch_size, -1, self.n_head_projection, self.key_size)
        q = self.apply_rope(q, product_ion_info_query)
        k = self.apply_rope(k, product_ion_info_key)

        #attention without FLASH Attention
        attn = torch.einsum('bnij,bmij->binm',q,k)/sqrt(self.key_size)
        attn = attn.masked_fill(~src_key_padding_mask, -float('inf')).softmax(dim=-1)
        postx = torch.einsum('binm,bmij->bnij',attn,v).flatten(2,3)
        
        ## Warning: SPDA cannot be used in pytorch 2.0 with CUDA 12.0 on A100
        ## Error Information: Both fused kernels do not support non-null attn_mask.
        #attention with FLASH Attention
        #postx = F.scaled_dot_product_attention(q.transpose(1,2), 
        #                                       k.transpose(1,2), 
        #                                       v.transpose(1,2), 
        #                                       attn_mask=src_key_padding_mask).transpose(1,2).flatten(-2)
        
        postx = self.output_layer(postx)
        x = self.dn(x, postx)
        return x
    
    @staticmethod
    def apply_rope(x, dis):
        dis_sin, dis_cos = dis.chunk(2,dim=-1)
        x0, x1 = x[..., 0::2], x[..., 1::2]
        return torch.concat([x0*dis_cos-x1*dis_sin,\
                             x1*dis_cos+x0*dis_sin], dim = -1)

class FFNGLU(nn.Module):
    def __init__(self, hidden_size: int, alpha: float, beta: float, dropout_rate: float):
        super().__init__()

        # 根据“GLU Variants Improve Transformer”，采用GEGLU结构做FFN.
        self.pre_ffn_gate = nn.Sequential(nn.Linear(hidden_size, 4*hidden_size, bias=False),
                                          nn.ReLU(inplace=True)
                                          )
        self.pre_ffn = nn.Linear(hidden_size, 4*hidden_size, bias=False)
        self.post_ffn = nn.Linear(4*hidden_size, hidden_size, bias=False)
        self.dn = DeepNorm(hidden_size, alpha, dropout_rate)

        #根据DeepNet，对初始化值做修正.
        nn.init.xavier_normal_(self.pre_ffn_gate[0].weight, gain=beta)
        nn.init.xavier_normal_(self.pre_ffn.weight, gain=beta)
        nn.init.xavier_normal_(self.post_ffn.weight, gain=beta)

    def forward(self, x):
        postx = self.post_ffn(self.pre_ffn_gate(x)*self.pre_ffn(x))
        x = self.dn(x, postx)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, max_charge, n_head_per_mass, key_size, alpha, beta, dropout_rate):
        super().__init__()
        self.mha = MultiHeadAttn(hidden_size,max_charge,n_head_per_mass,key_size,alpha,beta,dropout_rate)
        self.ffn = FFNGLU(hidden_size,alpha,beta,dropout_rate)

    def forward(self,x,product_ion_info_query,product_ion_info_key,src_key_padding_mask):
        x = checkpoint(self.mha,x,product_ion_info_query,product_ion_info_key,src_key_padding_mask)
        x = checkpoint(self.ffn,x)
        return x