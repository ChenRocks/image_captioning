import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from .im import ResNetFeatExtracotr
from .decoder import MultiLayerLSTMCells, LSTMAttnDecoder


class AttnImCap(nn.Module):
    def __init__(self, vsize, emb_dim, n_cell, n_layer):
        super().__init__()
        self._embedding = nn.Embedding(vsize, emb_dim, padding_idx=0)
        self._img_enc = ResNetFeatExtracotr(n_cell)
        self._dec_lstm = MultiLayerLSTMCells(n_cell+emb_dim, n_cell, n_layer)
        self._attn = nn.Linear(n_cell, n_cell)
        self._init_h = nn.Linear(n_cell, n_cell)
        self._init_c = nn.Linear(n_cell, n_cell)
        self._projection = nn.Linear(emb_dim+2*n_cell, emb_dim)
        self._decoder = LSTMAttnDecoder(
            self._embedding, self._dec_lstm, self._attn, self._projection)

    def set_embedding(self, embedding, finetune=False, oovs=[]):
        self._embedding.weight.data = embedding
        def oov_grad_hook(grad_input):
            """ hook to allow training only for randomly initialized oovs"""
            new_grad = grad_input.clone()
            new_grad.data.zero_()
            for i in oovs:
                new_grad.data[i, :] = grad_input.data[i, :]
            return new_grad

    def forward(self, img, input_):
        img_feat_map, init_states = self.encode(img)
        logit = self._decoder(img_feat_map, input_, init_states)
        return logit

    def encode(self, img):
        img_feat_map = self._img_enc(img)
        avg_attn = img_feat_map.mean(dim=3, keepdim=False
                                    ).mean(dim=2, keepdim=False)
        b = img.size(0)
        l = self._dec_lstm.num_layers
        d = self._dec_lstm.hidden_size
        init_states = (self._init_h(avg_attn).unsqueeze(0).expand(l, b, d),
                       self._init_c(avg_attn).unsqueeze(0).expand(l, b, d))
        return img_feat_map, init_states

    def decode(self, img, go, eos, max_len):
        img_feat_map, init_states = self.encode(img)
        tok = Variable(torch.LongTensor([go])).unsqueeze(1)
        if img.is_cuda:
            tok = tok.cuda(img.get_device())
        out_ids = []
        attns = []
        states = init_states
        for _ in range(max_len):
            out, states, attn = self._decoder.decode_step(
                tok, states, img_feat_map)
            if out.data[0, 0] == eos:
                break
            out_ids.append(out.data[0, 0])
            attns.append(attn.data[0])
            tok = out
        return out_ids, attns

    def batched_decode(self, img):
        pass

    # TODO
    def beamsearch(self, img):
        pass

    def batched_beamsearch(self, img):
        pass

