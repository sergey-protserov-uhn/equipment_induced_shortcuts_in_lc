import inspect
from itertools import product
from functools import reduce
from collections.abc import Iterator, Sequence, Mapping
import pickle
from warnings import warn
import numpy as np
import pandas as pd
import cv2
import skimage as si


def rescale_to_height(arr, target_height, resize_order, ensure_even_sizes=True):
    # ensure_even_sizes == True is consistent with ffmpeg
    if ensure_even_sizes and target_height % 2 != 0:
        raise ValueError(f"{ensure_even_sizes=}, but {target_height=} is not even")

    if arr.shape[0] == target_height and (
        (not ensure_even_sizes) or (arr.shape[1] % 2 == 0)
    ):
        return arr.copy()
    rescaling_factor = target_height / arr.shape[0]
    # Using int() rather than round() is consistent with ffmpeg
    target_width = int(arr.shape[1] * rescaling_factor)
    if ensure_even_sizes and target_width % 2 != 0:
        target_width += 1
    if resize_order == 0:
        anti_aliasing = False
    else:
        anti_aliasing = None
    arr = si.transform.resize(
        arr,
        (target_height, target_width),
        order=resize_order,
        anti_aliasing=anti_aliasing,
    )
    return arr


def pd_merge_from(dfs):
    if not isinstance(dfs, Iterator):
        dfs = iter(dfs)
    initial = next(dfs)
    return reduce(pd.merge, dfs, initial)


def all_equal(it):
    if not isinstance(it, Iterator):
        it = iter(it)
    try:
        initial = next(it)
    except StopIteration:
        return True
    return all(obj == initial for obj in it)


def cur_func_name():
    return inspect.stack()[1][3]


def image_arr2brightest_square(image_arr, size_frac=0.8, num_cands=(8, 18)):
    if len(image_arr.shape) == 3:
        if image_arr.shape[2] == 1:
            brightness_arr = image_arr[..., 0]
        else:
            brightness_arr = si.color.rgb2gray(image_arr)
    h, w = brightness_arr.shape
    size = int(min(h, w) * size_frac)
    cand_centers_h = np.unique(
        np.linspace(size // 2, h - (size - size // 2), num_cands[0]).round().astype(int)
    )
    cand_centers_w = np.unique(
        np.linspace(size // 2, w - (size - size // 2), num_cands[1]).round().astype(int)
    )
    num_cands = (len(cand_centers_h), len(cand_centers_w))
    square_brightnesses = np.zeros(num_cands, dtype=float)
    for (i, cand_center_h), (j, cand_center_w) in product(
        enumerate(cand_centers_h), enumerate(cand_centers_w)
    ):
        cand_square = brightness_arr[
            cand_center_h - size // 2 : cand_center_h + (size - size // 2),
            cand_center_w - size // 2 : cand_center_w + (size - size // 2),
        ]
        square_brightnesses[i, j] = cand_square.mean()
    brightest_center_ind = np.unravel_index(
        np.argmax(square_brightnesses), square_brightnesses.shape
    )
    brightest_center_h = cand_centers_h[brightest_center_ind[0]]
    brightest_center_w = cand_centers_w[brightest_center_ind[1]]
    brightest_slice = np.s_[
        brightest_center_h - size // 2 : brightest_center_h + (size - size // 2),
        brightest_center_w - size // 2 : brightest_center_w + (size - size // 2),
    ]
    return brightest_slice, image_arr[brightest_slice]


def unpack(o):
    if len(o) != 1:
        raise ValueError(f"{len(o)=} != 1")
    if isinstance(o, Sequence):
        return o[0]
    if isinstance(o, Mapping):
        return next(iter(o.items()))
    raise NotImplementedError(f"{type(o)=} is not supported")


class VideoFrames:
    def __init__(
        self,
        video_path,
        *,
        step_ms=None,
        step_frames=None,
        color_conversion=None,
        quiet=False,
    ):
        if (step_ms is None) == (step_frames is None):
            raise ValueError(
                f"Got {step_ms=}, {step_frames=}, but exactly one of these two values should be specified"
            )
        self.video_path = video_path
        self.step_ms = step_ms
        self.step_frames = step_frames
        self.color_conversion = color_conversion
        self.quiet = quiet

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.step_ms is not None:
            frames_per_step = self.fps * self.step_ms / 1000
            if frames_per_step != (frames_per_step_rounded := round(frames_per_step)):
                if not self.quiet:
                    warn(
                        f"Given {self.fps=} and {self.step_ms=}, {frames_per_step=} is not an integer. It will be rounded to {frames_per_step_rounded=}"
                    )
            self.frames_per_step = frames_per_step_rounded
        else:
            self.frames_per_step = self.step_frames
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.frames_per_step
        del self.frame_count
        del self.fps
        self.cap.release()
        del self.cap

    def __del__(self):
        if hasattr(self, "cap"):
            self.cap.release()

    def __len__(self):
        return self.frame_count // self.frames_per_step

    def frame_iter(self):
        i = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if i % self.frames_per_step == 0:
                if self.color_conversion is not None:
                    frame = cv2.cvtColor(frame, self.color_conversion)
                yield i, i * 1000 / self.fps, frame
            i += 1

    def __iter__(self):
        return self.frame_iter()


def pickle_(obj, file_path):
    with open(file_path, "wb") as out_file:
        pickle.dump(obj, out_file)


def unpickle(file_path):
    with open(file_path, "rb") as in_file:
        return pickle.load(in_file)
