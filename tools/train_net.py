# ------------------------------------------------------------------------
# Mostly a modified copy from timm (https://github.com/facebookresearch/SlowFast)
# ------------------------------------------------------------------------

"""
Train an/a image/video classification model.
"""

import pprint
from typing import Any

import numpy as np
import libs.models.losses as losses
import libs.models.optimizer as optim
import libs.utils.checkpoint as cu
import libs.utils.distributed as du
import libs.utils.logging as logging
import libs.utils.metrics as metrics
import libs.utils.misc as misc
import libs.visualization.tensorboard_vis as tb
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from libs.datasets import loader
from libs.models import build_model
from libs.utils.meters import TrainMeter, ValMeter

logger = logging.get_logger(__name__)


def train_epoch(
        train_loader,
        model,
        optimizer,
        train_meter,
        cur_epoch,
        cfg,
        writer=None
):
    """
    Perform the training process for one epoch.
    :param train_loader: training data loader.
    :param model: the model to train.
    :param optimizer:  the optimizer to perform optimization on the model's parameters.
    :param train_meter: training meters to log the training performance.
    :param cur_epoch: current epoch of training.
    :param cfg: configs. Details can be found in libs/config/defaults.py
    :param writer: TensorboardWriter object to writer Tensorboard log.
    """

    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    for cur_iter, (inputs, labels) in enumerate(train_loader):

        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    if isinstance(inputs[i], list):
                        for j in range(len(inputs[i])):
                            inputs[i][j] = inputs[i][j].cuda(non_blocking=True)
                    else:
                        inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        # Explicitly declare reduction to mean.
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

        # MixUP and Compute the loss.
        if cfg.TRAIN.MIXUP_ALPHA > 0:
            mixuper = optim.MixUper(
                mixup_alpha=cfg.TRAIN.MIXUP_ALPHA,
                criterion=loss_fun,
                use_cuda=True,
            )
            inputs1, inputs2 = inputs[0], inputs[1]
            inputs1, mixup_aux_a, mixup_aux_b, lam = mixuper.mixup_data(
                inputs1, labels
            )
            inputs = [inputs1, inputs2]
            predictions, _ = model(inputs)
            loss = mixuper.mixup_loss(predictions, mixup_aux_a, mixup_aux_b, lam)
        else:
            predictions, _ = model(inputs)
            loss = loss_fun(predictions, labels)

        # Check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()

        if cfg.SOLVER.CLIP_GRADIENT > 0.0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.SOLVER.CLIP_GRADIENT
            )

        # Update the parameters.
        optimizer.step()

        top1_err, top5_err = None, None
        if cfg.DATA.MULTI_LABEL:
            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                [loss] = du.all_reduce([loss])
            loss = loss.item()
        else:
            # Compute the errors.
            num_topks_correct = metrics.topks_correct(predictions, labels, (1, 5))
            top1_err, top5_err = [
                (1.0 - x / predictions.size(0)) * 100.0 for x in num_topks_correct
            ]

            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss, top1_err, top5_err = du.all_reduce(
                    [loss, top1_err, top5_err]
                )

            # Copy the stats from GPU to CPU (sync point).
            loss, top1_err, top5_err = (
                loss.item(),
                top1_err.item(),
                top5_err.item(),
            )

        # Update and log stats.
        train_meter.update_stats(
            top1_err,
            top5_err,
            loss,
            lr,
            inputs[0].size(0)
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
        )

        # Write to tensorboard format if available.
        if writer is not None:
            writer.add_scalars(
                {
                    "Train/loss": loss,
                    "Train/lr": lr,
                    "Train/Top1_err": top1_err,
                    "Train/Top5_err": top5_err,
                },
                global_step=data_size * cur_epoch + cur_iter,
            )

        train_meter.iter_toc()
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(
        val_loader,
        model,
        val_meter,
        cur_epoch,
        cfg,
        writer=None
):
    """
    Evaluate the model on the val set.
    :param val_loader: validation data loader.
    :param model: model to evaluate the performance.
    :param val_meter: meter instance to record and calculate the metrics.
    :param cur_epoch: number of the current epoch of training.
    :param cfg: configs. Details can be found in libs/config/defaults.py
    :param writer: TensorboardWriter object to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    if isinstance(inputs[i], list):
                        for j in range(len(inputs[i])):
                            inputs[i][j] = inputs[i][j].cuda(non_blocking=True)
                    else:
                        inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()

        val_meter.data_toc()

        predictions = model(inputs)

        if cfg.DATA.MULTI_LABEL:
            if cfg.NUM_GPUS > 1:
                predictions, labels = du.all_gather([predictions, labels])
        else:
            # Compute the errors.
            num_topks_correct = metrics.topks_correct(predictions, labels, (1, 5))

            # Combine the errors across the GPUs.
            top1_err, top5_err = [
                (1.0 - x / predictions.size(0)) * 100.0 for x in num_topks_correct
            ]
            if cfg.NUM_GPUS > 1:
                top1_err, top5_err = du.all_reduce([top1_err, top5_err])

            # Copy the errors from GPU to CPU (sync point).
            top1_err, top5_err = top1_err.item(), top5_err.item()

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(
                top1_err,
                top5_err,
                inputs[0].size(0)
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # Write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                    global_step=len(val_loader) * cur_epoch + cur_iter,
                )

        val_meter.update_predictions(predictions, labels)
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)

    # Write to tensorboard format if available.
    if writer is not None:
        all_predictions = [pred.clone().detach() for pred in val_meter.all_preds]
        all_labels = [label.clone().detach() for label in val_meter.all_labels]
        if cfg.NUM_GPUS:
            all_predictions = [pred.cpu() for pred in all_predictions]
            all_labels = [label.cpu() for label in all_labels]
        writer.plot_eval(
            preds=all_predictions,
            labels=all_labels,
            global_step=cur_epoch
        )

    val_meter.reset()


def calculate_and_update_precise_bn(
        data_loader,
        model, num_iters=200,
        use_gpu=True
):
    """
    Update the stats in bn layers by calculate the precise stats.
    :param data_loader: training data loader
    :param model: model to update the bn stats.
    :param num_iters: number of iterations to compute and update the bn stats.
    :param use_gpu: whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in data_loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            libs/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a model for many epochs on a train set and evaluate it on a val set.
    :param cfg: configs. Details can be found in libs/config/defaults.py.
    """

    du.init_distributed_training(cfg)  # Set up environment.
    np.random.seed(cfg.RNG_SEED)  # Set random seed.
    torch.manual_seed(cfg.RNG_SEED)

    logging.setup_logging(cfg.OUTPUT_DIR)  # Set up logging format.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the model and print its statistics.
    model = build_model(cfg)

    if du.is_master_proc() and cfg.LOG_MODEL_INFO and False:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)

    # Create the train and val data loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        train_epoch(
            train_loader,
            model,
            optimizer,
            train_meter,
            cur_epoch,
            cfg,
            writer
        )

        is_checkpoint_epoch = cu.is_checkpoint_epoch(cfg, cur_epoch)
        is_eval_epoch = misc.is_eval_epoch(cfg, cur_epoch)

        # Compute precise BN stats.
        if (
            (is_checkpoint_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkpoint_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg
            )

        # Evaluate the model on validation set.
        if is_eval_epoch:
            is_best_epoch = eval_epoch(
                val_loader,
                model,
                val_meter,
                cur_epoch,
                cfg,
                writer
            )

            if is_best_epoch:
                cu.save_checkpoint(
                    cfg.OUTPUT_DIR,
                    model,
                    optimizer,
                    cur_epoch,
                    cfg,
                    is_best_epoch,
                )

    if writer is not None:
        writer.close()
