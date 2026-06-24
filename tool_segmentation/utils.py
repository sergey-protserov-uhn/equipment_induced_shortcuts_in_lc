import json
import re
import sys
import tarfile
from contextlib import contextmanager, redirect_stdout
from datetime import datetime
from functools import wraps
from io import BytesIO, StringIO
from math import prod
from operator import itemgetter
from pathlib import Path
from warnings import warn

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import skimage as si
from joblib import Parallel, cpu_count, delayed
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm, trange

DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M:%S"

UPDATE_CODE_PATH_MSG = "This code path needs to be updated to be compatible with the rest of the codebase"


def rescale_to_height(arr, height, *, order, even_sizes=True):
    if even_sizes and (height % 2 != 0):
        msg = f"{even_sizes=}, but {height=}"
        raise ValueError(msg)

    cur_height, cur_width = arr.shape[:2]

    if (cur_height == height) and (
        (not even_sizes) or (cur_width % 2 == 0)
    ):
        return arr.copy()

    rescaling_factor = height / cur_height

    width = int(cur_width * rescaling_factor)
    if even_sizes and width % 2 != 0:
        width += 1

    if order == 0:
        anti_aliasing = False
    else:
        anti_aliasing = None

    return si.transform.resize(
        arr,
        (height, width),
        order=order,
        anti_aliasing=anti_aliasing,
    )


def binary_mask_arr2arr_with_removed_small_contours(
    mask_arr,
    retain_n_largest_contours,
):
    mask_arr = mask_arr.astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    contours_areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area == 0:
            warn(
                "Found contours with zero area, these will be discarded",
                stacklevel=2,
            )
            continue
        contours_areas.append((contour, area))
    return_mask = np.zeros_like(mask_arr)
    if contours_areas:
        contours_areas = sorted(
            contours_areas,
            key=itemgetter(1),
            reverse=True,
        )
        largest_contours = [
            contour
            for contour, _ in contours_areas[:retain_n_largest_contours]
        ]
        cv2.drawContours(return_mask, largest_contours, -1, 1, -1)
    return return_mask > 0


def class_arr2arr_with_removed_small_contours(
    class_arr,
    *,
    classes_to_process,
    retain_n_largest_contours,
    inpaint_with_class,
):
    class_arr = class_arr.copy()
    for class_idx in classes_to_process:
        class_mask = class_arr == class_idx
        filtered_mask = binary_mask_arr2arr_with_removed_small_contours(
            class_mask,
            retain_n_largest_contours,
        )
        class_arr[~filtered_mask & class_mask] = inpaint_with_class
    return class_arr


@contextmanager
def attr_assignment(o, name, value):
    if hasattr(o, name):
        saved_value = getattr(o, name)
        rm_attr = False
    else:
        rm_attr = True
    setattr(o, name, value)
    try:
        yield o
    finally:
        if rm_attr:
            delattr(o, name)
        else:
            setattr(o, name, saved_value)


def is_interactive():
    return hasattr(sys, "ps1")


def should_have_progress_bar():
    return sys.stderr.isatty() or is_interactive()


def maybe_trange(*args, **kwargs):
    if should_have_progress_bar():
        return trange(*args, **kwargs)
    return range(*args)


def maybe_tqdm(arg, **kwargs):
    if should_have_progress_bar():
        return tqdm(arg, **kwargs)
    return arg


@wraps(print)
def dprint(*args, **kwargs):
    dt = datetime.now()  # noqa: DTZ005
    date = dt.strftime(DATE_FORMAT)
    time = dt.strftime(TIME_FORMAT)
    print(
        f"On {date} at {time}:",
        *args,
        **kwargs,
    )


def train_val_test_split(
    *arrays,
    train_size,
    val_size,
    random_state_tv=None,
    random_state_vt=None,
    shuffle=True,
    stratify=None,
):
    if isinstance(train_size, int):
        train_size = int(train_size)
    elif isinstance(train_size, float):
        train_size = float(train_size)
    else:
        msg = f"{train_size=!r}"
        return TypeError(msg)
    size_type = type(train_size)
    if isinstance(val_size, size_type):
        val_size = size_type(val_size)
    else:
        msg = f"{val_size=!r}, but {size_type=}"
        raise TypeError(msg)
    arrays = list(arrays)
    if stratify is not None:
        arrays.append(stratify)
    train_valtest_arrays = train_test_split(
        *arrays,
        train_size=train_size,
        random_state=random_state_tv,
        shuffle=shuffle,
        stratify=stratify,
    )
    train_arrays = []
    valtest_arrays = []
    for i, a in enumerate(train_valtest_arrays):
        if i % 2 == 0:
            train_arrays.append(a)
        else:
            valtest_arrays.append(a)
    if stratify is not None:
        train_arrays.pop()
        stratify_valtest = valtest_arrays.pop()
    else:
        stratify_valtest = None
    if size_type is int:
        test_size = len(arrays[0]) - train_size - val_size
    elif size_type is float:
        test_size = (1 - train_size - val_size) / (1 - train_size)
    val_test_arrays = train_test_split(
        *valtest_arrays,
        test_size=test_size,
        random_state=random_state_vt,
        shuffle=shuffle,
        stratify=stratify_valtest,
    )
    val_arrays = []
    test_arrays = []
    for i, a in enumerate(val_test_arrays):
        if i % 2 == 0:
            val_arrays.append(a)
        else:
            test_arrays.append(a)
    return [
        array
        for tvt_arrays in zip(
            train_arrays, val_arrays, test_arrays, strict=True
        )
        for array in tvt_arrays
    ]


def get_info_from_job_dir(d):
    d = Path(d)
    with d.joinpath("cl_args.json").open() as in_file:
        cl_args = json.load(in_file)
    ckpt_name = max(d.joinpath("checkpoints/metric").glob("*.ckpt")).name
    epoch, metric_name, metric_val = re.findall(
        r"epoch=(\d+)_(\w+)=(.*)\.ckpt", ckpt_name
    )[0]
    epoch = int(epoch)
    metric_val = float(metric_val)
    return cl_args | {"epoch": epoch, metric_name: metric_val}


def save_image(image_arr, fp=None):
    if isinstance(fp, str):
        fp = Path(fp)
    if isinstance(fp, Path) and (fp.suffix != ".png"):
        msg = f"Saving as a PNG, but {fp=}"
        raise ValueError(msg)
    if fp is None:
        fp = BytesIO()
    image = Image.fromarray(si.util.img_as_ubyte(image_arr))
    image.save(fp, "PNG", optimize=True)
    if isinstance(fp, BytesIO):
        return fp
    return None


def drop_cols_with_same_val(df):
    return df.loc[:, df.nunique(dropna=False).gt(1)]


def process_metrics_csv(p):
    metrics_df = (
        pd.read_csv(p)
        .sort_values(by=["epoch", "step"])
        .drop(columns=["step"])
    )
    metrics_df = metrics_df[sorted(metrics_df.columns)]
    metrics_df_grouped = metrics_df.groupby("epoch")
    return metrics_df_grouped.first().sort_index().reset_index(drop=True)


def compute_metric(trues, preds, metric):
    par = Parallel(
        n_jobs=cpu_count(only_physical_cores=True),
        backend="threading",
        return_as="generator",
    )
    metric_vals = par(
        delayed(metric)(
            trues_[i],
            preds_[j],
        )
        for trues_, preds_ in zip(trues, preds, strict=True)
        for i in range(trues_.shape[0])
        for j in range(preds_.shape[0])
    )
    dims = (
        len(trues),
        trues[0].shape[0],
        preds[0].shape[0],
    )
    metric_vals = list(
        maybe_tqdm(metric_vals, total=prod(dims), leave=False)
    )
    metric_vals = np.array(metric_vals)
    return metric_vals.reshape(*dims, *metric_vals.shape[1:])


def create_annotated_heatmap(means, stds=None, *, precision=2, **kwargs):
    means = means[::-1]
    if stds is not None:
        stds = stds[::-1]
    annotation_text = []
    format_ = f".{precision}f"
    for i in range(means.shape[0]):
        annotation_row = []
        for j in range(means.shape[1]):
            cell_str = format(means[i, j], format_)
            if stds is not None:
                cell_str = f"{cell_str} ± {format(stds[i, j], format_)}"
            annotation_row.append(cell_str)
        annotation_text.append(annotation_row)
    kwargs = {
        "showscale": False,
        "x": list(range(means.shape[1])),
        "y": list(range(means.shape[0])),
    } | kwargs
    kwargs["y"] = kwargs["y"][::-1]
    fig = go.Figure(
        go.Heatmap(
            z=means,
            text=annotation_text,
            hoverinfo="none",
            texttemplate="%{text}",
            **(kwargs | {"x": None, "y": None}),
        )
    )
    fig.update_layout(
        xaxis={
            "tickmode": "array",
            "tickvals": list(range(len(kwargs["x"]))),
            "ticktext": kwargs["x"],
            "side": "top",
        }
    )
    fig.update_layout(
        yaxis={
            "tickmode": "array",
            "tickvals": list(range(len(kwargs["y"]))),
            "ticktext": kwargs["y"],
        }
    )
    return fig


def unpack(it):
    contents = list(it)
    if len(contents) != 1:
        msg = f"{len(contents)=}"
        raise ValueError(msg)
    return contents[0]


@contextmanager
def capture_stdout():
    sio = StringIO()
    with redirect_stdout(sio):
        yield sio


def pack_to_tar(tar_path, bios, paths):
    with tarfile.open(tar_path, "w") as tf:
        for bio, p in zip(bios, paths, strict=True):
            ti = tarfile.TarInfo(p)
            ti.size = len(bio.getbuffer())
            bio.seek(0)
            tf.addfile(ti, bio)
