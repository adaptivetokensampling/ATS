from os.path import basename
import subprocess
import argparse
from pathlib import Path

from joblib import Parallel, delayed
import pandas as pd


def video_process(video_file_path, dst_root_path, ext, fps=-1, size=240):
    if ext != video_file_path.suffix:
        print("ext: {} != suffix: {}".format(ext, video_file_path.suffix))
        return

    ffprobe_cmd = (
        "ffprobe -v error -select_streams v:0 "
        "-of default=noprint_wrappers=1:nokey=1 -show_entries "
        "stream=width,height,avg_frame_rate,duration"
    ).split()

    ffprobe_cmd.append(str(video_file_path))

    p = subprocess.run(ffprobe_cmd, capture_output=True)
    res = p.stdout.decode("utf-8").splitlines()
    if len(res) < 4:
        return

    dst_dir_path = dst_root_path / video_file_path.name

    if dst_dir_path.exists():
        return

    width = int(res[0])
    height = int(res[1])

    if width > height:
        vf_param = "scale=-1:{}".format(size)

    else:
        vf_param = "scale={}:-1".format(size)

    # if fps > 0:
    #     vf_param += 'minterpolate={}'.format(fps)

    ffmpeg_cmd = ["ffmpeg", "-i", str(video_file_path), "-vcodec", "mjpeg", "-vf", vf_param]
    ffmpeg_cmd += ["-threads", "1", "-an", "{}".format(dst_dir_path)]
    print(ffmpeg_cmd)
    subprocess.run(ffmpeg_cmd)
    print("\n")


def class_process(
    class_dir_path, dst_root_path, ext, video_id_df=None, fps=-1, size=240
):
    print(class_dir_path)
    if not class_dir_path.is_dir():
        print("Not dir!:{}".format(class_dir_path))
        return

    dst_class_path = dst_root_path / class_dir_path.name
    dst_class_path.mkdir(exist_ok=True)
    video_ids_per_class = []
    if video_id_df is not None:
        video_ids_per_class = list(
            video_id_df[video_id_df["label"] == class_dir_path.stem]["video_id"]
        )
    for video_file_path in sorted(class_dir_path.iterdir()):
        if (
            len(video_ids_per_class) != 0
            and video_file_path.stem not in video_ids_per_class
        ):
            continue
        video_process(video_file_path, dst_class_path, ext, fps, size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir_path", default=None, type=Path, help="Directory path of videos"
    )
    parser.add_argument(
        "dst_path", default=None, type=Path, help="Directory path of mjpeg videos"
    )
    parser.add_argument(
        "dataset",
        default="",
        type=str,
        help="Dataset name (kinetics | mit | ucf101 | hmdb51 | activitynet)",
    )
    parser.add_argument(
        "--class_list_path",
        default=None,
        type=Path,
        help="Directory path of the classes to use",
    )
    parser.add_argument(
        "--video_id_csv", default=None, type=Path, help="List of videos to process"
    )
    parser.add_argument(
        "--n_jobs", default=-1, type=int, help="Number of parallel jobs"
    )
    parser.add_argument(
        "--fps",
        default=-1,
        type=int,
        help=("Frame rates of output videos. " "-1 means original frame rates."),
    )
    parser.add_argument(
        "--size", default=256, type=int, help="Frame size of output videos."
    )
    parser.add_argument(
        "--vid_ext",
        default="mp4",
        type=str,
        help="Extension of videos, avi or mp4.",
        choices=["avi", "mp4"],
    )
    args = parser.parse_args()

    #     if args.dataset in ['mit', 'activitynet', 'kinetics']:
    #         ext = '.mp4'
    #     else:
    #         ext = '.avi'
    ext = f".{args.vid_ext}"
    if args.dataset == "activitynet":
        video_file_paths = [x for x in sorted(args.dir_path.iterdir())]
        status_list = Parallel(n_jobs=args.n_jobs, backend="threading")(
            delayed(video_process)(
                video_file_path, args.dst_path, ext, args.fps, args.size
            )
            for video_file_path in video_file_paths
        )
    else:
        video_id_df = None
        if args.video_id_csv:
            video_id_df = pd.read_csv(args.video_id_csv)
            class_name_list = list(video_id_df["label"].unique())
        elif args.class_list_path:
            class_name_list = [line.rstrip("\n") for line in open(args.class_list_path)]
        else:
            class_name_list = []

        if len(class_name_list) > 0:
            class_dir_paths = []
            for x in sorted(args.dir_path.iterdir()):
                class_name = basename(x)
                if class_name in class_name_list:
                    class_dir_paths.append(x)
        else:
            class_dir_paths = [x for x in sorted(args.dir_path.iterdir())]

        test_set_video_path = args.dir_path / "test"
        if test_set_video_path.exists():
            class_dir_paths.append(test_set_video_path)

        status_list = Parallel(n_jobs=args.n_jobs, backend="threading")(
            delayed(class_process)(
                class_dir_path, args.dst_path, ext, video_id_df, args.fps, args.size
            )
            for class_dir_path in class_dir_paths
        )
