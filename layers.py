import math
from turtle import forward
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model,
                 ):
        super(MultiHeadAttention, self).__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, head, mask=None):
        temp = q
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        bs, q_seq, d_model = q.size()
        _, k_seq, _ = k.size()
        dk = d_model//head

        q = q.view(bs, q_seq, head, dk).transpose(1, 2)
        k = k.view(bs, k_seq, head, dk).transpose(1, 2)
        v = v.view(bs, k_seq, head, dk).transpose(1, 2)

        att_score = q.matmul(k.transpose(-1, -2))/math.sqrt(dk)
        if mask is not None:
            att_score = mask(att_score)
        att_score = self.softmax(att_score)
        out = att_score.matmul(v)
        out = torch.cat([out[:, _, :, :] for _ in range(head)], dim=-1)

        return self.layer_norm(temp + out), att_score


class FeedForward(nn.Module):
    def __init__(self,
                 d_model,
                 feed_dim,
                 dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, feed_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(feed_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        temp = x
        x = self.relu(self.linear1(x))
        x = self.dropout(self.linear2(x))
        return self.layer_norm(x + temp)


class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 head,
                 feed_dim,
                 dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.head = head
        self.mha = MultiHeadAttention(d_model=d_model)
        self.feed_forward = FeedForward(d_model=d_model, feed_dim=feed_dim, dropout=dropout)

    def forward(self, q, k, v, mask=None):
        x, score = self.mha(q=q, k=k, v=v, head=self.head, mask=mask)
        x = self.feed_forward(x)
        return x, score

class DecoderLayer(nn.Module):
    def __init__(self, d_model,
                 head,
                 feed_dim,
                 dropout=0.1) -> None:
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model=d_model)
        self.mha2 = MultiHeadAttention(d_model=d_model)
        
        self.feed_dim = FeedForward(d_model=d_model, feed_dim=feed_dim, dropout=dropout)
        self.head = head 
    
    def forward(self, trg, memory, mask):
        x, _ = self.mha1(q=trg, k=trg, v=trg, head=self.head, mask=mask)
        x, _ = self.mha2(q=x, k=memory, v=memory, head=self.head)
        return self.feed_dim(x)
    
    
    
    
