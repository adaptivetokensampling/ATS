# ------------------------------------------------------------------------
# Mostly a modified copy from timm
# (https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)
# ------------------------------------------------------------------------

import logging
import math
from collections import OrderedDict
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ats_block import ATSBlock
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import build_model_with_cfg

from .transformer_block import Block
from functools import wraps
from ..build import MODEL_REGISTRY

_logger = logging.getLogger(__name__)


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    # patch models, imagenet21k (weights ported from official Google JAX impl)
    "vit_base_patch16_224_in21k": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth",
        num_classes=21843,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_base_patch16_384": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth",
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
    ),
    "vit_base_patch32_384": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth",
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
    ),
    "vit_base_patch32_224_in21k": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth",
        num_classes=21843,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_large_patch16_224_in21k": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth",
        num_classes=21843,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_large_patch32_224_in21k": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth",
        num_classes=21843,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
}


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """
    CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(
        self,
        backbone,
        img_size=224,
        feature_size=None,
        in_channels=3,
        embed_dim=768,
    ):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_channels, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, "feature_info"):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def initializer(init_func):
    @wraps(init_func)
    def new_init(self, cfg):
        init_func(
            self,
            cfg.VIT.IMG_SIZE,
            cfg.VIT.PATCH_SIZE,
            cfg.VIT.IN_CHANNELS,
            cfg.VIT.NUM_CLASSES,
            cfg.VIT.EMBED_DIM,
            cfg.VIT.DEPTH,
            cfg.VIT.NUM_HEADS,
            cfg.VIT.MLP_RATIO,
            cfg.VIT.QKV_BIAS,
            cfg.VIT.QK_SCALE,
            cfg.VIT.REPRESENTATION_SIZE,
            cfg.VIT.DROP_RATE,
            cfg.VIT.ATTN_DROP_RATE,
            cfg.VIT.DROP_PATH_RATE,
            cfg.VIT.HYBRID_BACKBONE,
            cfg.VIT.NORM_LAYER,
            cfg.VIT.ATS_BLOCKS,
            cfg.VIT.NUM_TOKENS,
            cfg.VIT.DROP_TOKENS,
        )

    return new_init


@MODEL_REGISTRY.register()
class ViT(nn.Module):
    """
    Vision Transformer
    A PyTorch implementation of An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    @initializer
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        hybrid_backbone=None,
        norm_layer=None,
        ats_blocks=[3, 4, 5, 6, 7, 8, 9, 10, 11],
        num_tokens=[197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197],
        drop_tokens=False,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_channels (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super(ViT, self).__init__()
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.img_size = img_size
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone,
                img_size=img_size,
                in_channels=in_channels,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=embed_dim,
            )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        control_flags = [True for _ in range(depth)]
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.ats_blocks = ats_blocks
        self.num_tokens = num_tokens

        self.blocks = []
        for i in range(depth):
            if i in self.ats_blocks:
                self.blocks.append(
                    ATSBlock(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        insert_control_point=control_flags[i],
                        drop_tokens=drop_tokens,
                    )
                )
            else:
                self.blocks.append(
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        insert_control_point=control_flags[i],
                    )
                )
        self.blocks = nn.ModuleList(self.blocks)
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(embed_dim, representation_size)),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        self._ref_num_tokens = None

    @staticmethod
    def _init_weights(m):
        """
        weight initialization
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens implementation from Phil Wang, thanks

        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        init_n = x.shape[1]
        policies = []
        policy = torch.ones(B, init_n, 1, dtype=x.dtype, device=x.device)
        sampler = torch.nonzero(policy)
        # idx = 0
        for idx, blk in enumerate(self.blocks):
            if idx in self.ats_blocks:
                x, policy = blk(
                    x=x,
                    n_tokens=self.num_tokens[idx],
                    policy=policy,
                    sampler=sampler,
                    n_ref_tokens=init_n,
                )
                # idx += 1
                # policies.append(policy)
            else:
                x = blk(x=x, policy=policy, sampler=sampler)
                # policies.append(policy)

        x = self.norm(x)[:, 0]
        x = self.pre_logits(x)
        return x, policies

    def forward(self, x):
        x, policies = self.forward_features(x)
        x = self.head(x)
        return x, policies


def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info("Resized position embedding: %s to %s", posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info("Position embedding grid-size from %s to %s", gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == "pos_embed" and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    default_cfg = default_cfgs[variant]
    default_num_classes = default_cfg["num_classes"]
    default_img_size = default_cfg["input_size"][-1]

    num_classes = kwargs.pop("num_classes", default_num_classes)
    img_size = kwargs.pop("img_size", default_img_size)
    repr_size = kwargs.pop("representation_size", None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        VisionTransformer,
        variant,
        pretrained,
        default_cfg=default_cfg,
        img_size=img_size,
        num_classes=num_classes,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs
    )
    model.default_cfg = default_cfg

    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch16_224_in21k(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=768,
        **kwargs
    )
    model = _create_vision_transformer(
        "vit_base_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch32_224_in21k(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=768,
        **kwargs
    )
    model = _create_vision_transformer(
        "vit_base_patch32_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_large_patch16_224_in21k(pretrained=False, **kwargs):
    """ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        representation_size=1024,
        **kwargs
    )
    model = _create_vision_transformer(
        "vit_large_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_large_patch32_224_in21k(pretrained=False, **kwargs):
    """ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=32,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        representation_size=1024,
        **kwargs
    )
    model = _create_vision_transformer(
        "vit_large_patch32_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch32_384(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch32_384", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch16_384", pretrained=pretrained, **model_kwargs
    )
    return model
