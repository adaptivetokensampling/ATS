# ------------------------------------------------------------------------
# Mostly a modified copy from
# "https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html#ImageFolder"
# ------------------------------------------------------------------------
import torch
import torchvision
import os
import os.path
from typing import Any
from PIL import Image
from functools import wraps
from torchvision.datasets.folder import DatasetFolder
from .build import DATASET_REGISTRY


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def create_transforms(
        cfg,
        split='train'
):
    """
    :param split: 'train' or 'val'
    :param cfg: configs. Details can be found in
            libs/config/defaults.py
    :return: image transformations
    """
    if split == 'train':
        crop_size = cfg.DATA.TRAIN_CROP_SIZE
    else:
        crop_size = cfg.DATA.TEST_CROP_SIZE

    size = int((256 / 224) * crop_size)
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=size, interpolation=3),
        torchvision.transforms.CenterCrop(crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    ])
    return transforms


def initializer(init_func):
    @wraps(init_func)
    def new_init(self, cfg, split):
        init_func(self, root=os.path.join(cfg.DATA.PATH_TO_DATA_DIR, split), transform=create_transforms(cfg, split))
    return new_init


def pil_loader(
        path: str
) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except OSError:
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


@DATASET_REGISTRY.register()
class ImageNet(DatasetFolder):
    @initializer
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=default_loader,
        is_valid_file=None,
    ):
        """
        ImageNet dataset. The code is taken from
        'https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html#ImageFolder'
        :param root: root directory path.
        :param transform: a function/transform that takes in an PIL image and returns a transformed version.
        E.g, transforms.RandomCrop
        :param target_transform: a function/transform that takes in the target and transforms it.
        :param loader: a function to load an image given its path.
        :param is_valid_file: a function that takes path of an Image file and check if the file is a valid file
        (used to check of corrupt files).
        """
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples


    #def __getitem__(self, item):
        #return torch.rand(3, 224, 224), 0

    #def __len__(self):
        #return 128




