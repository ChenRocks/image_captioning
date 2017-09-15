import torch
from torch import nn
from torch.nn import functional as F

from .attention import step_attention


class MultiLayerLSTMCells(nn.Module):
    """ stack multiple LSTM Cells"""
    def __init__(self, input_size, hidden_size, num_layers,
                 bias=True, dropout=0.0):
        super().__init__()
        cells = []
        cells.append(nn.LSTMCell(input_size, hidden_size, bias))
        for _ in range(num_layers-1):
            cells.append(nn.LSTMCell(hidden_size, hidden_size, bias))
        self._cells = nn.ModuleList(cells)
        self._dropout = dropout
        self.reset_parameters()

    def forward(self, input_, state):
        """
        Arguments:
            input_: Variable of FloatTensor (batch, input_size)
            states: tuple of the H, C LSTM states
                Variable of FloatTensor (num_layers, batch, hidden_size)
        Returns:
            LSTM states
            new_h: (num_layers, batch, hidden_size)
            new_c: (num_layers, batch, hidden_size)
        """
        hs = []
        cs = []
        for i, cell in enumerate(self._cells):
            s = (state[0][i, :, :], state[1][i, :, :])
            h, c = cell(input_, s)
            hs.append(h)
            cs.append(c)
            input_ = F.dropout(h, p=self._dropout, training=self.training)

        new_h = torch.stack(hs, dim=0)
        new_c = torch.stack(cs, dim=0)

        return new_h, new_c

    def reset_parameters(self):
        for cell in self._cells:
            # xavier initialization
            gate_size = self.hidden_size/4
            for weight in [cell.weight_ih, cell.weight_hh]:
                for w in torch.chunk(weight.data, 4, dim=0):
                    nn.init.xavier_normal(w)
            # forget_bias = 1
            for bias in [cell.bias_ih, cell.bias_hh]:
                torch.chunk(bias.data, 4, dim=0)[1].fill_(1)

    @property
    def hidden_size(self):
        return self._cells[0].hidden_size

    @property
    def input_size(self):
        return self._cells[0].input_size

    @property
    def num_layers(self):
        return len(self._cells)

    @property
    def bidirectional(self):
        return False


class LSTMAttnDecoder(object):
    def __init__(self, embedding, lstm_cell, attention, projection):
        super().__init__()
        self._embedding = embedding
        self._lstm_cell = lstm_cell
        self._attention = attention
        self._projection = projection

    def __call__(self, enc_img, input_, init_states):
        batch_size, max_len = input_.size()
        logits = []
        states = init_states
        for i in range(max_len):
            tok = input_[:, i:i+1]
            logit, states, _ = self._step(tok, states, enc_img)
            logits.append(logit)
        return torch.stack(logits, dim=1)

    def _step(self, tok, states, attention):
        h, c = states
        context, attn = step_attention(self._attention(h[-1]), attention)
        emb = self._embedding(tok).squeeze(1)
        h, c = self._lstm_cell(torch.cat([emb, context], dim=1), (h, c))
        out = self._projection(torch.cat([emb, context, h[-1]], dim=1))
        logit = torch.mm(out, self._embedding.weight.t())
        return logit, (h, c), attn

    def decode_step(self, tok, states, attention):
        logit, states, attn = self._step(tok, states, attention)
        out = logit.max(dim=1, keepdim=True)[1]
        return out, states, attn

