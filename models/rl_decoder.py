import torch
import numpy as np
import torch.nn as nn
from torch.cuda.amp import autocast
from gnova.modules.graphormer_graph_encoder import GraphormerGraphEncoder
from gnova_encoder import GNovaEncoder


# use graphormer_graph_encoder to do prediction
class rl_decoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = GNovaEncoder(cfg)
        self.decoder = GraphormerGraphEncoder(cfg)
