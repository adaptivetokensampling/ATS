# ------------------------------------------------------------------------
# Mostly a modified copy from timm
# (https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)
# ------------------------------------------------------------------------
import time

import numpy as np
import torch
from timm.models.layers import DropPath
from torch import Tensor
from torch import nn as nn
import torch.nn.functional as F
# from .cuda import gather_tokens, scatter_tokens


class AfterReconstruction(nn.Identity):
    def __init__(self, in_planes):
        super().__init__()
        self.in_planes = in_planes


class ALinear_CUDA(nn.Linear):
    """ """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        out_channels = self.weight.shape[0]
        self.out_zero_mask = nn.Parameter(
            torch.zeros(1, out_channels, 1), requires_grad=False
        )

    def forward(self, input: Tensor, mask: Tensor = None) -> Tensor:
        """

        :param input:
        :param mask:
        :return:
        """

        if mask is None:
            return F.linear(input, self.weight, self.bias)

        input = input.transpose(-2, -1).contiguous()
        B, D, N = input.shape

        # For Debug Only
        # mask[:, 10:110] = 0.0

        start = []

        start.append(time.time())
        active_position_list = torch.nonzero(mask.flatten()).squeeze(1).int()

        start.append(time.time())
        sampled_tokens = gather_tokens(input, active_position_list)

        start.append(time.time())
        sampled_tokens = sampled_tokens.transpose(-2, -1).squeeze().contiguous()
        # For Debug Only
        # print("{} / {}".format(sampled_tokens.shape[0], B*N))
        start.append(time.time())
        sampled_tokens = F.linear(sampled_tokens, self.weight, self.bias)

        start.append(time.time())
        # out_zero_mask = torch.zeros(flatten_input.shape[0], D_out).to(out.device)
        out_zero_mask = self.out_zero_mask.expand(B, -1, N).contiguous()

        start.append(time.time())
        out = scatter_tokens(sampled_tokens, out_zero_mask, active_position_list)

        start.append(time.time())
        out = out.transpose(-2, -1)

        start.append(time.time())

        start_gt = time.time()
        gt = F.linear(input.transpose(-2, -1).contiguous(), self.weight, self.bias)
        end_gt = time.time()

        timing = [(start[i] - start[i - 1]) for i in range(1, len(start))]

        total_time = sum(timing)

        print(
            "Linear Layer Timing: Adaptive Linear: {}  Linear: {}  Diff: {}".format(
                total_time, end_gt - start_gt, total_time - end_gt + start_gt
            )
        )

        timing_per = [
            str((start[i] - start[i - 1]) / total_time * 100)
            for i in range(1, len(start))
        ]
        str_timing = " | ".join(timing_per)
        print("Timing Details (%): " + str_timing)
        str_timing = " | ".join([str(t) for t in timing])
        print("Timing Details (s): " + str_timing)
        return out


class ALinear_PyTorch(nn.Linear):
    """ """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        out_channels = self.weight.shape[0]
        self.out_zero_mask = nn.Parameter(
            torch.zeros(1, out_channels), requires_grad=False
        )

    def forward(
        self, input: Tensor, mask: Tensor = None, sampler: Tensor = None
    ) -> Tensor:
        """

        :param input:
        :param mask:
        :param sampler:
        :return:
        """
        if mask is None:
            return F.linear(input, self.weight, self.bias)
        B, N, D = input.shape
        # timing = []
        D_out = self.weight.shape[0]

        # timing.append(time.time())
        # sampler = torch.nonzero(mask)
        out_mask_size = mask.sum(1).max().int()
        # if out_mask_size < 197:
        #    print("")
        #    pass
        # timing.append(time.time())
        sampler_out = sampler[:, 0] * out_mask_size + sampler[:, 1]
        sampler = sampler[:, 0] * N + sampler[:, 1]
        sampler_input = sampler.unsqueeze(-1).expand(-1, D)
        sampler_output = sampler_out.unsqueeze(-1).expand(-1, D_out)
        flatten_input = input.reshape(-1, D)

        # timing.append(time.time())
        sampled_input = torch.gather(flatten_input, 0, sampler_input)

        # timing.append(time.time())
        out = F.linear(sampled_input, self.weight, self.bias)

        # timing.append(time.time())
        out_zero_mask = self.out_zero_mask.expand(B * out_mask_size, -1)

        # timing.append(time.time())
        out = out_zero_mask.scatter(0, sampler_output, out, reduce="add").reshape(
            (B, out_mask_size, D_out)
        )
        policy = (
            out_zero_mask[:, 0]
            .scatter(0, sampler_out, 1, reduce="add")
            .reshape(B, out_mask_size, 1)
        )
        # timing.append(time.time())

        # gt_start = time.time()
        # gt = F.linear(input, self.weight, self.bias)
        # gt_end = time.time()

        # timing = [
        #    (timing[i] - timing[i - 1])
        #    for i in range(1, len(timing))
        # ]

        # total_time = sum(timing)

        # print(
        #    "Linear Layer Timing: Adaptive Linear: {}  Linear: {}  Diff: {}".format(
        #        total_time, gt_end - gt_start, total_time - gt_end + gt_start
        #    )
        # )

        # timing_per = [
        #    str(timing[i] / total_time * 100)
        #    for i in range(0, len(timing))
        # ]
        # str_timing = " | ".join(timing_per)
        # print("Timing Details (%): " + str_timing)
        # str_timing = " | ".join([str(t) for t in timing])
        # print("Timing Details (s): " + str_timing)

        return out, policy


class ALinear_Sparse(nn.Linear):
    def forward(self, input: Tensor, mask: Tensor, _) -> Tensor:
        B, N, D = input.shape
        # finput = (input * mask).flatten(0, 1)
        finput = input.flatten(0, 1)
        sp = finput.to_sparse_csr()
        out = sp.matmul(self.weight.transpose(-2, -1)) + self.bias.unsqueeze(0)
        out = out.reshape(B, N, -1)
        return out


class ALinear(nn.Linear):
    def forward(self, input: Tensor, mask: Tensor, _) -> Tensor:
        return super().forward(input)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ALinear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = ALinear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, policy: Tensor = None, sampler: Tensor = None) -> Tensor:
        x = self.fc1(x, policy, sampler)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x, policy, sampler)
        x = self.drop(x)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim**-0.5

        self.qkv = ALinear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = ALinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.n_segment = 8

    @staticmethod
    def softmax_with_policy(attn, policy, eps=1e-6):
        B, N, _ = policy.size()
        B, H, N, N = attn.size()
        attn_policy = policy.reshape(B, 1, 1, N)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(
            1, 1, N, N
        )
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(self, x, policy, sampler):

        B, N, C = x.shape

        qkv= self.qkv(x, policy, sampler)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if policy is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn, policy)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x, policy, sampler)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        insert_control_point=False,
    ):
        super().__init__()
        self.insert_control_point = insert_control_point
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, policy: Tensor = None, sampler: Tensor = None) -> Tensor:
        x = x + self.drop_path(
            self.attn(x=self.norm1(x), policy=policy, sampler=sampler)
        )
        if policy is not None:
            x = x * policy
        out = self.mlp(x=self.norm2(x), policy=policy, sampler=sampler)
        x = x + self.drop_path(out)
        if policy is not None:
            x = x * policy
        return x


def get_sinusoid_encoding(n_position, d_hid):
    """
    Sinusoid position encoding table
    """

    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
