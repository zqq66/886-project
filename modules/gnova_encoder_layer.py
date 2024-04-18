import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class DeepNorm(nn.Module):
    def __init__(self, normalized_shape, alpha, dropout_rate) -> None:
        super().__init__()
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout_rate)
        self.ln = nn.LayerNorm(normalized_shape)
    
    def forward(self, x, postx):
        return self.ln(x*self.alpha + self.dropout(postx))

class Relation(nn.Module):
    def __init__(self, 
                 hidden_size: int,
                 d_relation: int,
                 alpha: float,
                 beta: float,
                 dropout_rate: float):
        super().__init__()
        self.d_relation = d_relation
        self.hidden_size = hidden_size
        assert self.d_relation % 8 == 0
        assert self.hidden_size % self.d_relation == 0

        self.linear_q = nn.Linear(hidden_size, self.d_relation)
        self.linear_k = nn.Linear(hidden_size, self.d_relation)
        self.linear_predecessor = nn.Linear(hidden_size, self.d_relation)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.dist_embedding = nn.Embedding(100,self.d_relation)
        self.talking = nn.Linear(self.d_relation, self.d_relation, bias=False)
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.dn = DeepNorm(hidden_size, alpha, dropout_rate)
        self.predecessor_pad = nn.Parameter(torch.randn(self.d_relation))
        
        #根据DeepNet，对初始化值做修正.
        nn.init.xavier_normal_(self.linear_v.weight, gain=beta)
        nn.init.xavier_normal_(self.output_layer.weight, gain=beta)

    def forward(self, node, node_mass, dist, predecessors, rel_mask):
        node_num = node.size(1)
        q, k, predecessors_proj = self.linear_q(node), self.linear_k(node), self.linear_predecessor(node)
        # print('q', torch.isnan(q).any(), 'k', torch.isnan(k).any(), 'node_mass', torch.isnan(node_mass).any())
        q, k = self.apply_rope(q, node_mass), self.apply_rope(k, node_mass)
        # print('q', torch.isnan(q).any(), 'k', torch.isnan(k).any())
        dist = self.dist_embedding(dist)
        # print('dist', torch.isnan(dist).any())
        predecessors_mask = (predecessors==0)[...,0]
        predecessors = predecessors_proj.unsqueeze(2).expand(-1, -1, node_num, -1).gather(1, predecessors.expand(-1,-1,-1,self.d_relation))

        predecessors[predecessors_mask] = self.predecessor_pad.to(predecessors.dtype)

        v = self.linear_v(node).view(-1, node_num, self.d_relation, self.hidden_size//self.d_relation)
        # print('v', torch.isnan(v).any())
        relation = torch.einsum('bni,bmi->bnmi',q,k) + dist + predecessors
        # print('relation', torch.isnan(relation).any())
        relation = self.talking(relation)
        # print('relation-talk', torch.isnan(relation).any())
        relation = relation.masked_fill(~rel_mask, -float('inf')).softmax(dim=-2)
        # print('relation-masked', torch.isnan(relation).any())
        post_node = torch.einsum('bnmi,bmij->bnij',relation,v).flatten(2,3)
        # print('post_node', torch.isnan(post_node).any())
        post_node = self.output_layer(post_node)
        # print('post_node_out', torch.isnan(post_node).any())
        node = self.dn(node, post_node)
        # print('out', torch.isnan(node).any())
        return node
    
    @staticmethod
    def apply_rope(x, dis):
        dis_sin, dis_cos = dis.chunk(2,dim=-1)
        x0, x1 = x[..., 0::2], x[..., 1::2]
        return torch.concat([x0*dis_cos-x1*dis_sin,\
                             x1*dis_cos+x0*dis_sin], dim = -1)

class FFNGLU(nn.Module):
    def __init__(self, hidden_size: int, alpha: float, beta: float, dropout_rate: float):
        super().__init__()
        self.pre_ffn_gate = nn.Sequential(nn.Linear(hidden_size, 4*hidden_size, bias=False),
                                          nn.ReLU()
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

class GNovaEncoderLayer(nn.Module):
    def __init__(self, 
                 hidden_size: int,
                 d_relation: int,
                 alpha: float, 
                 beta: float, 
                 dropout_rate: float):

        super().__init__()
        self.relation = Relation(hidden_size, d_relation, alpha, beta, dropout_rate)
        self.ffn = FFNGLU(hidden_size, alpha, beta, dropout_rate)

    def forward(self, node, node_mass, dist, predecessors, rel_mask):
        #node = checkpoint(self.relation, node, node_mass, dist, predecessors, rel_mask)
        node = self.relation(node, node_mass, dist, predecessors, rel_mask)
        node = self.ffn(node)
        return node
