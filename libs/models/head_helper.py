

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn

from .transformers.transformer_block import Block


class VitHead(nn.Module):
    def __init__(self, embed_dim, cfg):
        super(VitHead, self).__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.temporal_encoder = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=cfg.TEMPORAL_HEAD.NUM_ATTENTION_HEADS,
                    attn_drop=cfg.TEMPORAL_HEAD.ATTENTION_PROBS_DROPOUT_PROB,
                    drop_path=cfg.TEMPORAL_HEAD.HIDDEN_DROPOUT_PROB,
                    drop=cfg.TEMPORAL_HEAD.HIDDEN_DROPOUT_PROB,
                    insert_control_point=False,
                )
                for _ in range(cfg.TEMPORAL_HEAD.NUM_HIDDEN_LAYERS)
            ]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(cfg.TEMPORAL_HEAD.HIDDEN_DIM),
            nn.Linear(cfg.TEMPORAL_HEAD.HIDDEN_DIM, cfg.TEMPORAL_HEAD.MLP_DIM),
            nn.GELU(),
            nn.Dropout(cfg.MODEL.DROPOUT_RATE),
            nn.Linear(cfg.TEMPORAL_HEAD.MLP_DIM, cfg.MODEL.NUM_CLASSES),
        )

    def forward(self, x, position_ids):
        # temporal encoder (Longformer)
        B, D, E = x.shape

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.temporal_encoder(x)
        # MLP head
        x = self.mlp_head(x[:, 0])
        return x
