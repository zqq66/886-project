import torch
import torch.nn as nn
from math import sqrt


class DeepNorm(nn.Module):
    def __init__(self, normalized_shape, alpha, dropout_rate) -> None:
        super().__init__()
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout_rate)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x, postx):
        # print('x, postx', x.shape, postx.shape)
        return self.ln(x * self.alpha + self.dropout(postx))


class MaskedMultiHeadAttn(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 n_head: int,
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
        head_size = hidden_size // n_head
        assert hidden_size % 8 == 0
        assert head_size % 8 == 0 and head_size <= 128
        assert hidden_size % n_head == 0

        self.n_head = n_head
        self.head_size = head_size
        self.hidden_size = hidden_size

        # 使用Pre Norm，降低训练难度
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.dn = DeepNorm(hidden_size, alpha, dropout_rate)

        nn.init.xavier_normal_(self.linear_v.weight, gain=beta)
        nn.init.xavier_normal_(self.output_layer.weight, gain=beta)

    def forward(self, tgt, tgt_mask, glycan_mass, glycan_query_mass):
        batch_size = tgt.size(0)
        q = self.linear_q(tgt).view(batch_size, -1, self.n_head, self.head_size)
        k = self.linear_k(tgt).view(batch_size, -1, self.n_head, self.head_size)
        v = self.linear_v(tgt).view(batch_size, -1, self.n_head, self.head_size)
        q = self.apply_rope(q, glycan_query_mass)
        k = self.apply_rope(k, glycan_mass)

        # attention without FLASH Attention
        attn = torch.einsum('bnij,bmij->binm', q, k) / sqrt(self.head_size)
        attn = attn.masked_fill(~tgt_mask, -float('inf')).softmax(dim=-1)
        postx = torch.einsum('binm,bmij->bnij', attn, v).flatten(2, 3)

        # attention with FLASH Attention
        # postx = F.scaled_dot_product_attention(q.transpose(1,2),
        #                                       k.transpose(1,2),
        #                                       v.transpose(1,2),
        #                                       attn_mask=tgt_mask).transpose(1,2).flatten(-2)

        postx = self.output_layer(postx)
        tgt = self.dn(tgt, postx)
        return tgt

    def inference_initialize(self, tgt, glycan_mass, query_glycan_mass):
        batch_size = tgt.size(0)
        q = self.linear_q(tgt).view(batch_size, -1, self.n_head, self.head_size)  # -1 is seqlen
        k = self.linear_k(tgt).view(batch_size, -1, self.n_head, self.head_size)
        v = self.linear_v(tgt).view(batch_size, -1, self.n_head, self.head_size)
        q = self.apply_rope(q, query_glycan_mass)  # 2, 1, 16, 128
        k = self.apply_rope(k, glycan_mass)  # 2, 1, 16, 128
        past_tgt = torch.concat([k, v], dim=-1)
        # print('q', q.shape, 'k', k.shape, 'v', v.shape)

        # attention without FLASH Attention
        attn = torch.einsum('bnij,bmij->binm', q, k) / sqrt(self.head_size)  # 2 16 1 1
        attn = attn.softmax(dim=-1)
        postx = torch.einsum('binm,bmij->bnij', attn, v).flatten(2, 3)  # 2 1 16 128
        # print('postx', postx.shape)

        # attention with FLASH Attention
        # postx = F.scaled_dot_product_attention(q.transpose(1,2),
        #                                       k.transpose(1,2),
        #                                       v.transpose(1,2)).transpose(1,2).flatten(-2)

        postx = self.output_layer(postx)
        tgt = self.dn(tgt, postx)
        return tgt, past_tgt

    def inference_step(self, tgt, glycan_mass, query_glycan_mass, past_tgt):
        batch_size = tgt.size(0)
        q = self.linear_q(tgt).view(batch_size, -1, self.n_head, self.head_size)
        k = self.linear_k(tgt).view(batch_size, -1, self.n_head, self.head_size)
        v = self.linear_v(tgt).view(batch_size, -1, self.n_head, self.head_size)
        q = self.apply_rope(q, query_glycan_mass)
        k = self.apply_rope(k, glycan_mass)
        k_past, v_past = past_tgt.split([self.head_size, self.head_size], dim=-1)
        k = torch.concat([k_past, k], dim=1)
        v = torch.concat([v_past, v], dim=1)
        past_tgt = torch.concat([k, v], dim=-1)

        # attention without FLASH Attention
        attn = torch.einsum('bnij,bmij->binm', q, k) / sqrt(self.head_size)
        attn = attn.softmax(dim=-1)

        postx = torch.einsum('binm,bmij->bnij', attn, v).flatten(2, 3)

        # attention with FLASH Attention
        # postx = F.scaled_dot_product_attention(q.transpose(1,2),
        #                                       k.transpose(1,2),
        #                                       v.transpose(1,2)).transpose(1,2).flatten(-2)

        postx = self.output_layer(postx)
        tgt = self.dn(tgt, postx)
        return tgt, past_tgt

    @staticmethod
    def apply_rope(x, dis):
        dis_sin, dis_cos = dis.chunk(2, dim=-1)
        # print('dis_sin, dis_cos', dis_sin.shape, dis_cos.shape)
        x0, x1 = x[..., 0::2], x[..., 1::2]
        # print('x0, x1', x0.shape, x1.shape)
        return torch.concat([x0 * dis_cos - x1 * dis_sin, \
                             x1 * dis_cos + x0 * dis_sin], dim=-1)


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
        assert key_size % 8 == 0 and key_size <= 128

        self.key_size = key_size
        self.n_head_projection = 2 * max_charge * n_head_per_mass

        # 使用Pre Norm，降低训练难度
        self.linear_q = nn.Linear(hidden_size, self.n_head_projection * key_size)
        self.linear_k = nn.Linear(hidden_size, self.n_head_projection * key_size)
        self.linear_v = nn.Linear(hidden_size, self.n_head_projection * key_size)
        self.output_layer = nn.Linear(self.n_head_projection * key_size, hidden_size)
        self.dn = DeepNorm(hidden_size, alpha, dropout_rate)

        nn.init.xavier_normal_(self.linear_v.weight, gain=beta)
        nn.init.xavier_normal_(self.output_layer.weight, gain=beta)

    def forward(self, tgt, mem, glycan_mass, node_mass, rel_mask):
        """_summary_

        Args:
            tgt (_type_): _description_
            mem_key_padding_mask (_type_): _description_
            pep_mass (_type_): _description_
            mem (_type_, optional): _description_. Defaults to None.
            peaks_moverz (_type_, optional): _description_. Defaults to None.
            past_mem (_type_, optional): _description_. Defaults to None.

        Raises:
            TypeError: _description_
            TypeError: _description_

        Returns:
            _type_: _description_
        """
        batch_size = tgt.size(0)
        q = self.linear_q(tgt).view(batch_size, -1, self.n_head_projection, self.key_size)
        k = self.linear_k(mem).view(batch_size, -1, self.n_head_projection, self.key_size)
        v = self.linear_v(mem).view(batch_size, -1, self.n_head_projection, self.key_size)
        # print('forward node_mass', node_mass.shape)
        # print('glycan_mass', glycan_mass.shape)
        q = self.apply_rope(q, glycan_mass)
        k = self.apply_rope(k, node_mass)

        # attention without FLASH Attention
        attn = torch.einsum('bnij,bmij->binm', q, k) / sqrt(self.key_size)
        attn = attn.masked_fill(~rel_mask, -float('inf')).softmax(dim=-1)
        # print('attn', attn.shape)
        postx = torch.einsum('binm,bmij->bnij', attn, v).flatten(2, 3)

        # attention with FLASH Attention
        # postx = F.scaled_dot_product_attention(q.transpose(1,2),
        #                                       k.transpose(1,2),
        #                                       v.transpose(1,2),
        #                                       attn_mask=mem_key_padding_mask).transpose(1,2).flatten(-2)

        postx = self.output_layer(postx)
        # print('tgt', tgt.shape, v.shape)

        tgt = self.dn(tgt, postx)
        return tgt

    def inference_initialize(self, tgt, rel_mask, mem, glycan_mass, node_mass):
        batch_size = tgt.size(0)
        # print('mha, tgt', tgt.shape)

        q = self.linear_q(tgt).view(batch_size, -1, self.n_head_projection, self.key_size)
        k = self.linear_k(mem).view(batch_size, -1, self.n_head_projection, self.key_size)
        v = self.linear_v(mem).view(batch_size, -1, self.n_head_projection, self.key_size)
        # print('q', q.shape)
        # print('mha, q', q.shape)
        q = self.apply_rope(q, glycan_mass)  # [2, 2, 16, 64]
        # print('glycan_mass', glycan_mass.shape)
        # print('q', q.shape, 'k', k.shape, 'node_mass', node_mass.shape)
        k = self.apply_rope(k, node_mass)  # [2, 314, 16, 64]
        past_mem = torch.concat([k, v], dim=-1)

        # attention without FLASH Attention
        attn = torch.einsum('bnij,bmij->binm', q, k) / sqrt(self.key_size)
        # print('attn', attn.shape, 'rel_mask', rel_mask.shape)
        attn = attn.masked_fill(~rel_mask, -float('inf')).softmax(dim=-1)
        postx = torch.einsum('binm,bmij->bnij', attn, v).flatten(2, 3)

        # attention with FLASH Attention
        # postx = F.scaled_dot_product_attention(q.transpose(1,2),
        #                                       k.transpose(1,2),
        #                                       v.transpose(1,2),
        #                                       attn_mask=mem_key_padding_mask).transpose(1,2).flatten(-2)

        postx = self.output_layer(postx)
        tgt = self.dn(tgt, postx)
        return tgt, past_mem

    def inference_step(self, tgt, rel_mask, past_mem, glycan_mass):
        batch_size = tgt.size(0)
        q = self.linear_q(tgt).view(batch_size, -1, self.n_head_projection, self.key_size)
        # print('q',q.shape, glycan_mass.shape)

        q = self.apply_rope(q, glycan_mass)
        k, v = past_mem.split([self.key_size, self.key_size], dim=-1)

        # attention without FLASH Attention
        attn = torch.einsum('bnij,bmij->binm', q, k) / sqrt(self.key_size)
        attn = attn.masked_fill(~rel_mask, -float('inf')).softmax(dim=-1)
        postx = torch.einsum('binm,bmij->bnij', attn, v).flatten(2, 3)

        # attention with FLASH Attention
        # postx = F.scaled_dot_product_attention(q.transpose(1,2),
        #                                       k.transpose(1,2),
        #                                       v.transpose(1,2),
        #                                       attn_mask=mem_key_padding_mask).transpose(1,2).flatten(-2)

        postx = postx.reshape(batch_size, -1, self.n_head_projection * self.key_size)
        postx = self.output_layer(postx)
        tgt = self.dn(tgt, postx)
        return tgt

    @staticmethod
    def apply_rope(x, dis):
        dis_sin, dis_cos = dis.chunk(2, dim=-1)
        x0, x1 = x[..., 0::2], x[..., 1::2]
        return torch.concat([x0 * dis_cos - x1 * dis_sin, \
                             x1 * dis_cos + x0 * dis_sin], dim=-1)


class FFN(nn.Module):
    def __init__(self, hidden_size: int, alpha: float, beta: float, dropout_rate: float):
        super().__init__()

        # 根据“GLU Variants Improve Transformer”，采用GEGLU结构做FFN.
        self.ffn = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(4 * hidden_size, hidden_size, bias=False))
        self.dn = DeepNorm(hidden_size, alpha, dropout_rate)

        # 根据DeepNet，对初始化值做修正.
        nn.init.xavier_normal_(self.ffn[0].weight, gain=beta)
        nn.init.xavier_normal_(self.ffn[-1].weight, gain=beta)

    def forward(self, x):
        postx = self.ffn(x)
        x = self.dn(x, postx)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, n_head, max_charge,
                 n_head_per_mass, key_size, alpha, beta, dropout_rate):
        super().__init__()
        self.mmha = MaskedMultiHeadAttn(hidden_size=hidden_size,
                                        n_head=n_head,
                                        alpha=alpha,
                                        beta=beta,
                                        dropout_rate=dropout_rate)

        self.mha = MultiHeadAttn(hidden_size=hidden_size,
                                 max_charge=max_charge,
                                 n_head_per_mass=n_head_per_mass,
                                 key_size=key_size,
                                 alpha=alpha,
                                 beta=beta,
                                 dropout_rate=dropout_rate)

        self.ffn = FFN(hidden_size=hidden_size,
                       alpha=alpha,
                       beta=beta,
                       dropout_rate=dropout_rate)

    def forward(self, tgt, *, tgt_mask, rel_mask, glycan_mass, glycan_query_mass, mem, node_mass,
                glycan_crossattn_mass):
        tgt = self.mmha(tgt, tgt_mask, glycan_mass, glycan_query_mass)
        # print('forward', tgt[0])
        tgt = self.mha(tgt,
                       mem=mem,
                       glycan_mass=glycan_crossattn_mass,
                       node_mass=node_mass,
                       rel_mask=rel_mask)
        tgt = self.ffn(tgt)

        return tgt

    def inference_initialize(self, tgt, *, rel_mask, glycan_mass, query_glycan_mass, mem, node_mass,
                             glycan_crossattn_mass):
        tgt, past_tgt = self.mmha.inference_initialize(tgt, glycan_mass=glycan_mass,
                                                       query_glycan_mass=query_glycan_mass)
        # print('self', tgt.shape)
        tgt, past_mem = self.mha.inference_initialize(tgt,
                                                      mem=mem,
                                                      glycan_mass=glycan_crossattn_mass,
                                                      node_mass=node_mass,
                                                      rel_mask=rel_mask)
        # print('cross', tgt.shape)
        # add similarity network to find parent index for tgt
        tgt = self.ffn(tgt)
        # print('inference_initialize', tgt[0])

        return tgt, past_tgt, past_mem

    def inference_step(self, tgt, *, past_tgt, rel_mask, glycan_mass, past_mem, glycan_crossattn_mass,
                       query_glycan_mass):

        tgt, past_tgt = self.mmha.inference_step(tgt, past_tgt=past_tgt, glycan_mass=glycan_mass,
                                                 query_glycan_mass=query_glycan_mass)
        tgt = self.mha.inference_step(tgt,
                                      glycan_mass=glycan_crossattn_mass,
                                      past_mem=past_mem,
                                      rel_mask=rel_mask)
        tgt = self.ffn(tgt)
        # print('inference_step', tgt[0])
        return tgt, past_tgt