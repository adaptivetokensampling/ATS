from libs.utils.misc import launch_job
from libs.utils.parser import load_config, parse_args
from test_net import test
from train_net import train
from visualization import visualize
import sys
sys.path.append(".")

"""
Wrapper to train and test a video classification model.
"""


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    # Perform train.
    if cfg.TRAIN.ENABLE:
        launch_job(
            cfg=cfg,
            init_method=args.init_method,
            func=train
        )

    # Perform test.
    if cfg.TEST.ENABLE:
        launch_job(
            cfg=cfg,
            init_method=args.init_method,
            func=test
        )

    # Perform model visualization.
    if cfg.TENSORBOARD.ENABLE and (
        cfg.TENSORBOARD.MODEL_VIS.ENABLE
        or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
    ):
        launch_job(cfg=cfg, init_method=args.init_method, func=visualize)


if __name__ == "__main__":
    main()
