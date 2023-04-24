import math
from .layers import EncoderLayer, DecoderLayer
import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, d_model,
                 head,
                 feed_dim,
                 n_layers, 
                 dropout=0.1):
        super().__init__()
        self.encoder = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                    head=head,
                                                    feed_dim=feed_dim,
                                                    dropout=dropout) for _ in range(n_layers)])
    
    def forward(self, x, mask=None):
        for layer in self.encoder:
            x = layer(x)
        
        return x         


class Decoder(nn.Module):
    def __init__(self, d_model,
                 head,
                 feed_dim,
                 n_layers, 
                 dropout=0.1):
        super().__init__()
        self.encoder = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                    head=head,
                                                    feed_dim=feed_dim,
                                                    dropout=dropout) for _ in range(n_layers)])
    
    def forward(self, trg, memory, mask):
        for layer in self.encoder:
            x = layer(trg=trg, memory=memory, mask=mask)
        
        return x    

