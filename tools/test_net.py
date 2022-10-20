

"""
Test an image classification model.
"""

import os
import pickle
import time

import numpy as np
import libs.utils.distributed as du
import libs.utils.logging as logging
import libs.visualization.tensorboard_vis as tb
import torch
from fvcore.common.file_io import PathManager
from libs.datasets import loader
from libs.models import build_model
from libs.utils.meters import TestMeter
from libs.models.transformers.vit import checkpoint_filter_fn


logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    :param test_loader: test data loader
    :param model: the pretrained video model to test
    :param test_meter:  testing meters to log and ensemble the testing
            results
    :param cfg: configs. Details can be found in
            libs/config/defaults.py
    :param writer: TensorboardWriter object
            to writer Tensorboard log.
    :return: test_meter
    """

    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()
    num_tokens = []
    total = 0
    steps = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # if cur_iter == 9:
        #    break
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

        test_meter.data_toc()

        # Perform the forward pass.
        start.record()
        predictions, policies = model(inputs)
        end.record()
        torch.cuda.synchronize()

        if cur_iter > 6:
            time_diff = start.elapsed_time(end)/1000.0
            total += time_diff
            steps += 1
            # print("TIME: " + str(total / steps))

        num_tokens.append([p.sum(1).max().cpu().numpy() for p in policies])

        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            predictions, labels = du.all_gather([predictions, labels])
        if cfg.NUM_GPUS:
            predictions = predictions.cpu()
            labels = labels.cpu()

        test_meter.iter_toc()

        # Update and log stats.
        test_meter.update_stats(
            predictions.detach(),
            labels.detach(),
            torch.arange(0, labels.shape[0]) + (cur_iter * cfg.TEST.BATCH_SIZE),
        )
        test_meter.log_iter_stats(cur_iter)
        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    all_predictions = test_meter.video_preds.clone().detach()
    all_labels = test_meter.video_labels
    if cfg.NUM_GPUS:
        all_predictions = all_predictions.cpu()
        all_labels = all_labels.cpu()
    if writer is not None:
        writer.plot_eval(preds=all_predictions, labels=all_labels)

    if cfg.TEST.SAVE_RESULTS_PATH != "":
        save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

        with PathManager.open(save_path, "wb") as f:
            pickle.dump([all_labels, all_labels], f)

        logger.info("Successfully saved prediction results to {}".format(save_path))
    test_meter.finalize_metrics()

    avg_tokens = num_tokens[0]
    for i in range(1, len(num_tokens)):
        for j in range(len(avg_tokens)):
            avg_tokens[j] += num_tokens[i][j]

    #str_avg_tokens = [str(t / len(num_tokens)) for t in avg_tokens]
    #str_avg_tokens = ",".join(str_avg_tokens)
    #print("AVG TOKENS:", str_avg_tokens)

    return test_meter


def test(cfg):
    """
    Perform testing on a pretrained image model.
    :param cfg: configs. Details can be found in
            libs/config/defaults.py.
    """

    # Set up environment.
    du.init_distributed_training(cfg)

    # Set random seed.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Set up logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the model and print its statistics.
    model = build_model(cfg)

    # Load weights.
    ckpt = torch.load(cfg.TEST.CHECKPOINT_FILE_PATH, map_location="cpu")
    ckpt = checkpoint_filter_fn(model=model, state_dict=ckpt)
    logger.info("Warning: model.load_state_dict set to strict=False")
    model.load_state_dict(ckpt, strict=False)

    # Create test loader.
    test_loader = loader.construct_loader(cfg, "val")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    assert (
        len(test_loader.dataset)
        % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
        == 0
    )
    # Create meters for testing.
    test_meter = TestMeter(
        len(test_loader.dataset)
        // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
        cfg.MODEL.NUM_CLASSES,
        len(test_loader),
        cfg.DATA.MULTI_LABEL,
        cfg.DATA.ENSEMBLE_METHOD,
    )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)

    if writer is not None:
        writer.close()
