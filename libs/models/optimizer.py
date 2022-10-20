

"""Optimizer."""

import numpy as np
import libs.utils.lr_policy as lr_policy
import torch


def _get_optim_policies_TSN(model, cfg):
    first_conv_weight = []
    first_conv_bias = []
    normal_weight = []
    normal_bias = []
    lr5_weight = []
    lr10_bias = []
    bn = []
    custom_ops = []

    conv_cnt = 0
    bn_cnt = 0
    for m in model.modules():
        if (
            isinstance(m, torch.nn.Conv2d)
            or isinstance(m, torch.nn.Conv1d)
            or isinstance(m, torch.nn.Conv3d)
        ):
            ps = list(m.parameters())
            conv_cnt += 1
            if conv_cnt == 1:
                first_conv_weight.append(ps[0])
                if len(ps) == 2:
                    first_conv_bias.append(ps[1])
            else:
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
        elif isinstance(m, torch.nn.Linear):
            ps = list(m.parameters())
            normal_weight.append(ps[0])
            if len(ps) == 2:
                normal_bias.append(ps[1])

        elif isinstance(m, torch.nn.BatchNorm2d):
            bn_cnt += 1
            # later BN's are frozen
            if not cfg.TSN.PARTIAL_BN or bn_cnt == 1:
                bn.extend(list(m.parameters()))
        elif isinstance(m, torch.nn.BatchNorm3d):
            bn_cnt += 1
            # later BN's are frozen
            if not cfg.TSN.PARTIAL_BN or bn_cnt == 1:
                bn.extend(list(m.parameters()))
        elif len(m._modules) == 0:
            if len(list(m.parameters())) > 0:
                raise ValueError(
                    "New atomic module type: {}. Need to give it a learning policy".format(
                        type(m)
                    )
                )

    return [
        {
            "params": first_conv_weight,
            "lr_mult": 1,
            "decay_mult": 1,
            "name": "first_conv_weight",
        },
        {
            "params": first_conv_bias,
            "lr_mult": 2,
            "decay_mult": 0,
            "name": "first_conv_bias",
        },
        {
            "params": normal_weight,
            "lr_mult": 1,
            "decay_mult": 1,
            "name": "normal_weight",
        },
        {
            "params": normal_bias,
            "lr_mult": 2,
            "decay_mult": 0,
            "name": "normal_bias",
        },
        {"params": bn, "lr_mult": 1, "decay_mult": 0, "name": "BN scale/shift"},
        {
            "params": custom_ops,
            "lr_mult": 1,
            "decay_mult": 1,
            "name": "custom_ops",
        },
        # for fc
        {
            "params": lr5_weight,
            "lr_mult": 5,
            "decay_mult": 1,
            "name": "lr5_weight",
        },
        {
            "params": lr10_bias,
            "lr_mult": 10,
            "decay_mult": 0,
            "name": "lr10_bias",
        },
    ]


def construct_optimizer(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    if cfg.MODEL.ARCH == "xvit" and not (
        "deit" in cfg.XVIT.BASE_MODEL
        or "vit" in cfg.XVIT.BASE_MODEL
        or "swin" in cfg.XVIT.BASE_MODEL
    ):
        optim_params = _get_optim_policies_TSN(model, cfg)
    else:
        # Batchnorm parameters.
        bn_params = []
        # Non-batchnorm parameters.
        non_bn_parameters = []
        # Classifier parameters
        classifier_parameters_w = []
        classifier_parameters_b = []
        for m_name, m in model.named_modules():
            is_bn = isinstance(m, torch.nn.modules.batchnorm._NormBase)
            if "new_fc" in m_name or "head" in m_name:
                for p_name, p in m.named_parameters(recurse=False):
                    if "bias" in p_name:
                        classifier_parameters_b.append(p)
                    else:
                        classifier_parameters_w.append(p)
            else:
                for p in m.parameters(recurse=False):
                    if is_bn:
                        bn_params.append(p)
                    else:
                        non_bn_parameters.append(p)

        # Apply different weight decay to Batchnorm and non-batchnorm parameters.
        # In Caffe2 classification codebase the weight decay for batchnorm is 0.0.
        # Having a different weight decay on batchnorm might cause a performance
        # drop.
        optim_params = [
            {"params": bn_params, "weight_decay": cfg.BN.WEIGHT_DECAY},
            {
                "params": non_bn_parameters,
                "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": classifier_parameters_w,
                "lr": cfg.SOLVER.BASE_LR,
                "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": classifier_parameters_b,
                "lr": cfg.SOLVER.BASE_LR,
                "weight_decay": cfg.BN.WEIGHT_DECAY,
            },
        ]
        print(classifier_parameters_w[0].size(), len(classifier_parameters_w))
        # Check all parameters will be passed into optimizer.
        assert len(list(model.parameters())) == len(non_bn_parameters) + len(
            bn_params
        ) + len(classifier_parameters_b) + len(
            classifier_parameters_w
        ), "parameter size does not match: {} + {} != {}".format(
            len(non_bn_parameters),
            len(bn_params),
            len(list(model.parameters())),
        )

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adamw":
        return torch.optim.AdamW(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )


def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decays.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


class MixUper:
    def __init__(
        self,
        mixup_alpha,
        criterion,
        mixup_off_epochs=0,
        num_epochs=None,
        use_cuda=True,
    ):
        assert (
            mixup_off_epochs == 0 or num_epochs is not None
        ), "If mixup_off_epochs is greater than 0, then must pass num_epochs"
        self.mixup_alpha = mixup_alpha
        self.mixup_off_epochs = mixup_off_epochs
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.use_cuda = use_cuda

    def _do_mix(self, x, index, lam):

        # new data
        mixed_x = lam * x + (1 - lam) * x[index, :]

        return mixed_x

    def mixup_data(self, x, y, epoch=None):
        # sample lam
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        if self.mixup_off_epochs > 0 and epoch >= (
            self.num_epochs - self.mixup_off_epoch
        ):
            lam = 1

        # re-order index vector
        if isinstance(x, list):
            batch_size = x[0].size()[0]
        else:
            batch_size = x.size()[0]

        if self.use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        # new labels
        y_a, y_b = y, y[index]

        if isinstance(x, list):
            mixed_x = []
            for inp in x:
                inp_mu = self._do_mix(inp, index, lam)
                mixed_x.append(inp_mu)
        else:
            mixed_x = self._do_mix(x, index, lam)

        return mixed_x, y_a, y_b, lam

    def mixup_loss(self, pred, y_a, y_b, lam):
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(
            pred, y_b
        )
