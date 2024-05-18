import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F

from ..utils.generic_utils import to_cuda
from ..utils.constants import VERY_SMALL_NUMBER, INF
from .attention import *
# from .quant import quantize

# bit_list = [8, 8, 8, 8] # [4, 4, 2, 2]
class GF(nn.Module):
    def __init__(self, hidden_size):
        super(GF, self).__init__()
        self.fc_z = nn.Linear(4 * 100, hidden_size, bias=True)

    def forward(self, h_state, input):
        # h_state = quantize(h_state, num_bits=bit_list[0], dequantize=True)
        # input = quantize(input, num_bits=bit_list[1], dequantize=True)
        z = torch.sigmoid(self.fc_z(torch.cat([h_state, input, h_state * input, h_state - input], -1)))
        h_state = (1 - z) * h_state + z * input
        return h_state


class GS(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(GS, self).__init__()
        self.linear_z = nn.Linear(200, hidden_size, bias=False)
        self.linear_r = nn.Linear(200, hidden_size, bias=False)
        self.linear_t = nn.Linear(200, hidden_size, bias=False)

    def forward(self, h_state, input):
        # h_state = quantize(h_state, num_bits=bit_list[0], dequantize=True)
        # input = quantize(input, num_bits=bit_list[1], dequantize=True)
        z = torch.sigmoid(self.linear_z(torch.cat([h_state, input], -1)))
        r = torch.sigmoid(self.linear_r(torch.cat([h_state, input], -1)))
        t = torch.tanh(self.linear_t(torch.cat([r * h_state, input], -1)))
        h_state = (1 - z) * h_state + z * t
        return h_state


def dropout(x, drop_prob, shared_axes=[], training=False):
    """
    Apply dropout to input tensor.
    Parameters
    ----------
    input_tensor: ``torch.FloatTensor``
        A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)``
    Returns
    -------
    output: ``torch.FloatTensor``
        A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)`` with dropout applied.
    """
    if drop_prob == 0 or drop_prob == None or (not training):
        return x

    sz = list(x.size())
    for i in shared_axes:
        sz[i] = 1
    mask = x.new(*sz).bernoulli_(1. - drop_prob).div_(1. - drop_prob)
    mask = mask.expand_as(x)
    return x * mask
