import torch
from torch import nn
from torch.nn import functional as F

def step_attention(query, attention):
    """query [B, D], attention: [B, D, W, H]"""
    bs, d, w, h = attention.size()
    attention = attention.contiguous().view(bs, d, -1)
    score = torch.matmul(query.unsqueeze(1), attention).squeeze(1)
    normalized_score = F.softmax(score)
    context = torch.matmul(normalized_score.unsqueeze(1),
                           attention.transpose(1, 2)).squeeze(1)
    return context, normalized_score.contiguous().view(bs, w, h)
