import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformers import VisionTransformer

###################


class SegmentConsensus(torch.nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(SegmentConsensus, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == "avg":
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == "identity":
            output = input_tensor
        else:
            output = None

        return output


class ConsensusModule(torch.nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = (
            consensus_type if consensus_type != "rnn" else "identity"
        )
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)


##############


class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div

    def forward(self, x):
        nt, num_heads, d, c = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, num_heads, d, c)
        fold = c * num_heads // self.fold_div

        x = (
            x.permute(0, 1, 2, 4, 3)
            .contiguous()
            .view(n_batch, self.n_segment, num_heads * c, d)
        )
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]  # shift right
        out[:, :, 2 * fold :] = x[:, :, 2 * fold :]  # not shift

        out = (
            out.view(n_batch, self.n_segment, num_heads, c, d)
            .permute(0, 1, 2, 4, 3)
            .contiguous()
        )

        return out.view(nt, num_heads, d, c)


def make_temporal_shift(net, n_segment, n_div=8, locations_list=[]):
    n_segment_list = [n_segment] * 20
    assert n_segment_list[-1] > 0

    counter = 0
    for idx, block in enumerate(net.blocks):
        if idx in locations_list:
            net.blocks[idx].attn.control_point_query = TemporalShift(
                net.blocks[idx].attn.control_point_query,
                n_segment=n_segment_list[counter + 2],
                n_div=n_div,
            )
            net.blocks[idx].attn.control_point_value = TemporalShift(
                net.blocks[idx].attn.control_point_value,
                n_segment=n_segment_list[counter + 2],
                n_div=n_div,
            )
            counter += 1
