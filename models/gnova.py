import torch.nn as nn
from .gnova_encoder import GNovaEncoder

class GNova(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = GNovaEncoder(cfg)
        self.output_linear = nn.Linear(cfg.model.hidden_size,1)
    
    def forward(self, *, encoder_input):
        graph_node = self.encoder(**encoder_input)
        tgt = self.output_linear(graph_node)
        return tgt
