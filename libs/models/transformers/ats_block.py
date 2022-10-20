import time

from torch import Tensor
from .transformer_block import Attention, Mlp
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class AdaptiveTokenSampler(Attention):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        drop_tokens=False,
    ):
        super(AdaptiveTokenSampler, self).__init__(
            dim,
            num_heads,
            qkv_bias,
            qk_scale,
            attn_drop,
            proj_drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.out_zero_mask = nn.Parameter(torch.zeros(1, dim), requires_grad=False)
        self.drop_tokens = drop_tokens

    @staticmethod
    def get_unique_indices(indices: Tensor, max_value: int) -> Tensor:
        """
        :param indices: indices of the tokens to be sampled
        :param max_value: maximum number of the tokens to be sampled
        :return: unique indices of the tokens to be sampled
        """
        sorted_indices = torch.sort(indices, dim=1)[0]

        shift_left = F.pad(sorted_indices[:, 1:], (0, 1), value=1.0)
        unique_indices = torch.where(
            (shift_left - sorted_indices) == 0,
            max_value * torch.ones_like(indices),
            sorted_indices,
        )
        unique_indices = torch.sort(unique_indices, dim=1)[0]
        return unique_indices

    @staticmethod
    def create_ys(normalized_cdf: Tensor, n_tokens: int) -> Tensor:
        """
        Sample uniformly from y-axis.
        """

        B = normalized_cdf.shape[0]
        # epsilon = (1 / (n_tokens - 1)) / 2
        ys = (
            torch.linspace(
                start=0,
                end=1.0,
                steps=n_tokens - 1,
                device=normalized_cdf.device,
            )
            .unsqueeze(0)
            .repeat(B, 1)
        )
        ys_start = (
            torch.min(normalized_cdf + (normalized_cdf == 0).float() * 1e8, dim=1)[0]
            .unsqueeze(-1)
            .expand_as(ys)
        )
        steps = (
            torch.range(0, n_tokens - 2, device=normalized_cdf.device)
            .unsqueeze(0)
            .expand_as(ys_start)
        )
        ys = ys_start + (((ys * (n_tokens - 2)) - ys_start * steps) / (n_tokens - 2))

        return ys

    @staticmethod
    def score_assignment_step(attn: Tensor, v: Tensor) -> (Tensor, Tensor):
        """
        Token Score Assignment Step.
        :param attn: attention matrix
        :param v: values
        :return: sorted significance scores and their corresponding indices
        """

        B, H, _, _ = attn.shape
        C = v.shape[3] * H
        v_norm = torch.linalg.norm(
            v.transpose(1, 2).reshape(B, attn.shape[2], C), ord=2, dim=2
        )  # value norm of size [B x T]
        significance_score = attn[:, :, 0].sum(
            dim=1
        )  # attention weights of CLS token of size [B x T]
        significance_score = significance_score * v_norm  # [B x T]
        significance_score = significance_score[:, 1:]  # [B x T-1]

        significance_score = significance_score / significance_score.sum(
            dim=1, keepdim=True
        )  # [B x T-1]
        sorted_scores, sorted_indices = torch.sort(
            significance_score, descending=False, dim=1
        )

        return sorted_scores, sorted_indices

    def inverse_transform_sampling(
        self,
        sorted_scores: Tensor,
        sorted_indices: Tensor,
        attn: Tensor,
        n_tokens: int,
        raw_x: Tensor,
        n_ref_tokens: int,
    ) -> (Tensor, Tensor):
        """
        Sample tokens based on their significance scores.
        """
        B, N, C = raw_x.shape

        cdf = torch.cumsum(sorted_scores, dim=1)  # [B x T-1]

        normalized_cdf = (  # normalized cdf
            cdf - cdf.min(dim=1)[0].unsqueeze(dim=1)
        ) / ((cdf.max(dim=1)[0] - cdf.min(dim=1)[0]) / 1.0).unsqueeze(dim=1)

        ys = self.create_ys(normalized_cdf, n_ref_tokens).unsqueeze(
            dim=2
        )  # sampled values from y-axis of size [B, n-1, 1]
        normalized_cdf = normalized_cdf.unsqueeze(dim=1)  # [B, 1, N - 1]

        # expanded_ys = torch.Tensor.expand(ys, (B, n_tokens - 1, N - 1))
        expanded_ys = torch.Tensor.expand(ys, (B, ys.shape[1], ys.shape[1]))
        diff_tokens = ys.shape[1] - (N - 1)
        tokens_to_pick_ind = torch.min(
            torch.abs(expanded_ys - F.pad(normalized_cdf, (diff_tokens, 0))),
            dim=2,
        )[
            1
        ]  # [B x n-1]

        # Offsetting token indices
        tokens_to_pick_ind = tokens_to_pick_ind - diff_tokens

        # Sort attention matrix and add CLS weights.
        attn_sorted = torch.gather(
            attn[:, :, 1:],
            2,
            sorted_indices.unsqueeze(1)
            .unsqueeze(-1)
            .expand(B, self.num_heads, N - 1, N),
        )  # [B x h x T-1 x T]

        attn_tmp = F.pad(attn_sorted, (0, 0, 0, 1), value=0.0)  # [B x h x T x T]

        # # Sort tokens and add CLS token.
        raw_x_tmp = torch.gather(
            raw_x[:, 1:], 1, sorted_indices.unsqueeze(-1).expand(B, N - 1, C)
        )
        raw_x_tmp = F.pad(raw_x_tmp, (0, 0, 0, 1), value=0.0)  # [B x n x C]

        unique_indices = self.get_unique_indices(
            indices=tokens_to_pick_ind, max_value=N - 1
        )[:, : N - 1]

        # Prune the attention matrix and input tokens.
        attn_tmp = torch.gather(
            attn_tmp,
            2,
            unique_indices.unsqueeze(1)
            .unsqueeze(3)
            .expand(B, self.num_heads, n_tokens - 1, N),
        )
        raw_x_tmp = torch.gather(
            raw_x_tmp, 1, unique_indices.unsqueeze(2).expand(B, n_tokens - 1, C)
        )

        attn_tmp = torch.cat([attn[:, :, 0:1], attn_tmp], dim=2)
        raw_x_tmp = torch.cat([raw_x[:, 0:1], raw_x_tmp], dim=1)

        policy = (unique_indices != (N - 1)).unsqueeze(-1).float()
        policy = F.pad(policy, (0, 0, 1, 0), value=1.0)
        selected_x = raw_x_tmp
        attn = attn_tmp

        sampler = torch.nonzero(policy)

        return selected_x, attn, policy, sampler

    def forward(
        self,
        x: Tensor,
        policy: Tensor,
        sampler: Tensor,
        n_tokens: float,
        raw_x: Tensor,
        n_ref_tokens: int,
    ):
        B, N, C = x.shape

        if isinstance(N, Tensor):
            N = N.cpu().item()

        if n_tokens > N:  # Number of tokens to be sampled can't be larger than N.
            n_tokens = N
        if n_tokens <= 1.0:  # When n_tokens is a ratio.
            n_tokens = n_tokens * N
        if n_tokens < 8:  # Number of tokens to be sampled can't be less than 8.
            n_tokens = 8

        n_tokens = round(n_tokens)
        if N < n_tokens:
            n_tokens = N

        qkv = self.qkv(x, policy, sampler)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        qkv = qkv * policy.unsqueeze(0).unsqueeze(
            2
        )  # Get rid of previously removed tokens.
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn_no_softmax = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax_with_policy(attn_no_softmax, policy)  # [B x H x T x T]

        # --------------------------
        # Token Score Assignment
        # --------------------------

        sorted_scores, sorted_indices = self.score_assignment_step(attn, v)

        # --------------------------
        # Inverse Transform Sampling
        # --------------------------

        selected_x, attn, policy, sampler = self.inverse_transform_sampling(
            sorted_scores, sorted_indices, attn, n_tokens, raw_x, n_ref_tokens
        )

        x = (attn @ v).transpose(1, 2).reshape(B, attn.shape[2], C)

        # Pruning
        if self.drop_tokens:
            out_mask_size = policy.sum(1).max().int()

            sampler_out = sampler[:, 0] * out_mask_size + sampler[:, 1]
            sampler = sampler[:, 0] * n_tokens + sampler[:, 1]
            sampler_input = sampler.unsqueeze(-1).expand(-1, C)
            sampler_output = sampler_out.unsqueeze(-1).expand(-1, C)
            flatten_x = x.reshape(-1, C)
            flatten_selected_x = selected_x.reshape(-1, C)

            x_prunned = torch.gather(flatten_x, 0, sampler_input)
            selected_x_prunned = torch.gather(flatten_selected_x, 0, sampler_input)

            out_zero_mask = self.out_zero_mask.expand(B * out_mask_size, -1)

            x = out_zero_mask.scatter(
                0, sampler_output, x_prunned, reduce="add"
            ).reshape((B, out_mask_size, C))
            selected_x = out_zero_mask.scatter(
                0, sampler_output, selected_x_prunned, reduce="add"
            ).reshape((B, out_mask_size, C))

            policy = (
                out_zero_mask[:, 0]
                .scatter(0, sampler_out, 1, reduce="add")
                .reshape(B, out_mask_size, 1)
            )

        x = self.proj(x, policy, sampler)
        x = x * policy
        x = self.proj_drop(x)
        return x, selected_x, policy, sampler


class ATSBlock(nn.Module):
    """
    Transformer Block + ATS
    """

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
        drop_tokens=False,
    ):
        super().__init__()
        self.insert_control_point = insert_control_point
        self.norm1 = norm_layer(dim)

        self.attn = AdaptiveTokenSampler(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            drop_path=drop_path,
            drop_tokens=drop_tokens,
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

    def forward(
        self,
        x,
        n_tokens,
        policy: Tensor = None,
        sampler: Tensor = None,
        n_ref_tokens: int = 197,
    ):
        x_out, selected_x, policy, sampler = self.attn(
            x=self.norm1(x),
            policy=policy,
            sampler=sampler,
            n_tokens=n_tokens,
            raw_x=x,
            n_ref_tokens=n_ref_tokens,
        )
        x = selected_x + self.drop_path(x_out)
        x = x * policy
        out = self.mlp(x=self.norm2(x), policy=policy, sampler=sampler)
        x = x + self.drop_path(out)
        x = x * policy
        return x, policy
