import io
import random

import numpy as np
import torch
from PIL import Image


def temporal_sampling(
    hdf5_video, hdf5_video_key, start_idx, end_idx, num_samples, video_length
):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, video_length - 1).long().tolist()
    try:
        data = hdf5_video[hdf5_video_key][index]
    except:
        data = [hdf5_video[hdf5_video_key][i] for i in index]
    try:
        frames = []
        for raw_frame in data:
            frames.append(
                np.asarray(Image.open(io.BytesIO(raw_frame)).convert("RGB"))
            )
    except:
        print(f"{hdf5_video_key}, {start_idx}, {end_idx}")

    frames = torch.as_tensor(np.stack(frames))
    return frames, index


def temporal_sampling_uniform(
    hdf5_video, hdf5_video_key, mode, clip_idx, num_samples, video_length
):
    temporal_sample_index = clip_idx

    seg_size = float(video_length - 1) / num_samples
    seq = []
    for i in range(num_samples):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        if mode == "train":
            seq.append(random.randint(start, end))
        elif mode == "val":
            seq.append((start + end) // 2)
        elif mode == "test":
            if temporal_sample_index == 0:
                seq.append((start + end) // 2)
            elif temporal_sample_index == 1:
                seq.append(start)
            elif temporal_sample_index == 2:
                seq.append(end)

    data = [hdf5_video[hdf5_video_key][i] for i in seq]

    try:
        frames = []
        for raw_frame in data:
            frames.append(
                np.asarray(Image.open(io.BytesIO(raw_frame)).convert("RGB"))
            )
    except:
        pass

    return torch.as_tensor(np.stack(frames)), seq


def get_start_end_idx(video_size, clip_size, clip_idx, num_clips):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx


def decode(
    hdf5_video,
    hdf5_video_key,
    sampling_rate,
    num_frames,
    clip_idx=-1,
    num_clips=10,
    video_meta=None,
    target_fps=30,
    max_spatial_scale=0,
    mode="train",
):
    """
    Decode the video and perform temporal sampling.
    Args:
        hdf5_video (container): hdf5 video.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips clips, and select the
            clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly
            sample from the given video.
        video_meta (dict): a dict contains VideoMetaData. Details can be find
            at `pytorch/vision/torchvision/io/_video_opt.py`.
        target_fps (int): the input video may have different fps, convert it to
            the target video fps before frame sampling.
        max_spatial_scale (int): keep the aspect ratio and resize the frame so
            that shorter edge size is max_spatial_scale. Only used in
            `torchvision` backend.
    Returns:
        frames (tensor): decoded frames from the video.
    """
    assert clip_idx >= -1, "Not valid clip_idx {}".format(clip_idx)

    # Perform selective decoding.
    if "fps" in video_meta:
        sampling_fps = (
            num_frames * sampling_rate * video_meta["fps"] / target_fps
        )
    else:
        sampling_fps = num_frames * sampling_rate

    start_idx, end_idx = get_start_end_idx(
        video_meta["num_frames"], sampling_fps, clip_idx, num_clips
    )

    # Perform temporal sampling from the decoded video.
    frames, frames_index = temporal_sampling(
        hdf5_video,
        hdf5_video_key,
        start_idx,
        end_idx,
        num_frames,
        video_meta["num_frames"],
    )
    # frames, frames_index = temporal_sampling_uniform(hdf5_video, hdf5_video_key, mode, clip_idx, num_frames, video_meta['num_frames'])

    return frames, frames_index
