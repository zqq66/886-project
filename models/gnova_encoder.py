import torch
import torch.nn as nn
from math import pi
from gnova.modules import GNovaEncoderLayer
from torch.utils.checkpoint import checkpoint

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim, lambda_max=1e4, lambda_min=1e-5) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        base = lambda_max/(2*pi)
        scale = lambda_min/lambda_max
        self.div_term = nn.Parameter(base*scale**(torch.arange(0, dim, 2)/dim), requires_grad=False)

    def forward(self, mass_position):
        pe_sin = torch.sin(mass_position.unsqueeze(dim=-1) / self.div_term)
        pe_cos = torch.cos(mass_position.unsqueeze(dim=-1) / self.div_term)
        return torch.concat([pe_sin, pe_cos],dim=-1).float()

class GNovaEncoder(nn.Module):
    def __init__(self, cfg):
        """_summary_

        Args:
            cfg (_type_): _description_
            bin_classification (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.node_feature_proj = nn.Linear(9, cfg.model.hidden_size)
        self.node_sourceion_embedding = nn.Embedding(20, cfg.model.hidden_size, padding_idx=0)
        self.node_mass_embedding = SinusoidalPositionEmbedding(cfg.model.d_relation)
        
        self.genova_encoder_layers = GNovaEncoderLayer(hidden_size = cfg.model.hidden_size, d_relation = cfg.model.d_relation,
                              alpha = (2*cfg.model.num_layers)**0.25, beta = (8*cfg.model.num_layers)**-0.25, 
                              dropout_rate = cfg.model.dropout_rate)
        self.cfg = cfg
        
    def forward(self, node_feature, node_sourceion, node_mass, dist, predecessors, rel_mask):
        node = self.node_feature_proj(node_feature) + self.node_sourceion_embedding(node_sourceion)
        node_mass = self.node_mass_embedding(node_mass)
        for _ in range(self.cfg.model.num_layers):
            node = self.genova_encoder_layers(node, node_mass, dist, predecessors, rel_mask)

        return node