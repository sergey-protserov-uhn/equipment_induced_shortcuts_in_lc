import argparse
import json
import sys
from ast import literal_eval
from datetime import timedelta
from functools import partial
from pathlib import Path
from shutil import rmtree
from sys import stderr, stdout

import albumentations as A
import lightning as L
import numpy as np
import pandas as pd
import skimage as si
import torch as t

from .data import SemanticSegmentationDataModule
from .metrics import NumPyMetricAdapter, f1_score
from .modules import SemanticSegmentationModule
from .unet import UNet
from .utils import (
    dprint,
    maybe_tqdm,
    process_metrics_csv,
    should_have_progress_bar,
)

if __name__ == "__main__":
    stdout.reconfigure(line_buffering=True)
    stderr.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--model_class")
    parser.add_argument("--model_kwargs", type=literal_eval)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--initial_lr", type=float, default=1e-6)
    parser.add_argument("--max_lr", type=float, default=1e-2)
    parser.add_argument("--min_lr", type=float, default=1e-10)
    parser.add_argument("--label_smoothing_alpha", type=float, default=0)
    parser.add_argument("--disable_augmentations", action="store_true")
    args = parser.parse_args()

    args_json = json.dumps(vars(args))
    dprint("args_json=")
    print(args_json)
    with Path("cl_args.json").open("w", encoding="utf-8") as out_file:
        out_file.write(args_json + "\n")

    df = pd.read_csv("processed_metadata_split_for_endoscapes.csv")
    df = df.sample(frac=0.01)
    image_arrs = []
    mask_arrs = []
    for subset in maybe_tqdm(("train", "val")):
        subset_df = df[df["subset"] == subset]
        subset_image_arrs = []
        subset_mask_arrs = []
        for row in maybe_tqdm(
            subset_df.itertuples(), total=len(subset_df), leave=False
        ):
            mask_arr = si.util.img_as_ubyte(si.io.imread(row.mask_path))
            mask_arr = np.where(
                mask_arr.max(axis=-1) > 100, 255, 0
            ).astype(np.uint8)
            if not np.any(mask_arr):
                continue
            subset_image_arrs.append(si.io.imread(row.image_path))
            mask_arr = np.stack(
                (mask_arr, 255 - mask_arr), axis=-1, dtype=np.uint8
            )
            subset_mask_arrs.append(mask_arr)
        image_arrs.append(subset_image_arrs)
        mask_arrs.append(subset_mask_arrs)

    if not args.disable_augmentations:
        train_geometric_augmentations = [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.ThinPlateSpline((0.01, 0.05)),
            A.Affine(
                scale=(0.8, 1.25),
                translate_percent=(-0.25, 0.25),
                rotate=(-180, 180),
                keep_ratio=True,
                balanced_scale=True,
            ),
        ]
        train_pixel_augmentations = [
            A.ColorJitter(hue=0.05),
            A.GaussNoise((0.01, 0.05)),
        ]
    else:
        train_geometric_augmentations = []
        train_pixel_augmentations = []
    train_pixel_transform = A.Sequential(train_pixel_augmentations, p=1)

    train_geometric_transform = A.Sequential(
        train_geometric_augmentations, p=1
    )
    val_geometric_transform = None

    mask_padding_val = [0, 1]
    n_input_channels = 3
    n_classes = 2
    metric_labels = (0, 1)

    dm = SemanticSegmentationDataModule(
        train_image_arrs=image_arrs[0],
        train_mask_arrs=mask_arrs[0],
        val_image_arrs=image_arrs[1],
        val_mask_arrs=mask_arrs[1],
        train_image_and_mask_transform=train_geometric_transform,
        train_image_transform=train_pixel_transform,
        val_image_and_mask_transform=val_geometric_transform,
        mask_padding_val=mask_padding_val,
        standardize=True,
        batch_size=args.batch_size,
    )

    if args.model_class == "UNet":
        model_class = UNet
        model_args = []
        n_input_channels_ = n_input_channels
        n_output_channels = n_classes
        model_kwargs = args.model_kwargs | {
            "n_input_channels": n_input_channels_,
            "n_output_channels": n_output_channels,
            "pool_class": t.nn.MaxPool2d,
            "activation_class": t.nn.LeakyReLU,
            "activation_args": [],
            "activation_kwargs": {},
        }

    label_smoothing_alpha = (
        args.label_smoothing_alpha * n_classes / (n_classes - 1)
    )

    metric_class = NumPyMetricAdapter
    metric_args = []
    metric_kwargs = {
        "metric_func": partial(f1_score, labels=metric_labels)
    }
    metric_name = "f1"

    metric_to_monitor = f"val_{metric_name}"

    metric_model_ckpt_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath="checkpoints/metric",
        filename=f"{{epoch:05d}}_{{{metric_to_monitor}:.7f}}",
        monitor=metric_to_monitor,
        save_weights_only=True,
        mode="max",
    )
    epoch_model_ckpt_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath="checkpoints/epoch",
        filename="{epoch:05d}",
    )

    timer_callback = L.pytorch.callbacks.Timer()

    logger = L.pytorch.loggers.CSVLogger("./", flush_logs_every_n_steps=1)

    callbacks = [
        timer_callback,
        metric_model_ckpt_callback,
        epoch_model_ckpt_callback,
    ]

    trainer = L.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=args.epochs,
        log_every_n_steps=1,
        enable_progress_bar=should_have_progress_bar(),
    )
    with trainer.init_module():
        module = SemanticSegmentationModule(
            model_class=model_class,
            model_args=model_args,
            model_kwargs=model_kwargs,
            initial_lr=args.initial_lr,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            label_smoothing_alpha=label_smoothing_alpha,
            metric_class=metric_class,
            metric_args=metric_args,
            metric_kwargs=metric_kwargs,
            metric_name=metric_name,
        )

    ckpt_paths = sorted(
        Path(epoch_model_ckpt_callback.dirpath).glob("*.ckpt"),
        reverse=True,
    )
    ckpt_path = None
    if ckpt_paths:
        for i, ckpt_path_ in enumerate(ckpt_paths):  # noqa: B007
            try:
                t.load(
                    ckpt_path_,
                    map_location="cpu",
                    weights_only=False,
                )
                ckpt_path = ckpt_path_
                dprint(f"Resuming from {ckpt_path=}")
                break
            except Exception:  # noqa: BLE001
                ckpt_path_.unlink()
        for ckpt_path_ in ckpt_paths[i + 1 :]:
            ckpt_path_.unlink()

    trainer.fit(module, datamodule=dm, ckpt_path=ckpt_path)

    dprint("Time elapsed:")
    for k, v in timer_callback.state_dict()["time_elapsed"].items():
        if v != 0:
            print(f"{k}: {timedelta(seconds=v)}")
    metric_ckpt_paths = sorted(
        Path(metric_model_ckpt_callback.dirpath).glob("*.ckpt"),
        reverse=True,
    )
    best_checkpoint_path = metric_ckpt_paths[0].as_posix()
    dprint("best_checkpoint_path=")
    print(best_checkpoint_path)
    for ckpt_path in metric_ckpt_paths[1:]:
        ckpt_path.unlink()
    rmtree(epoch_model_ckpt_callback.dirpath)
    metrics_paths = sorted(
        Path("lightning_logs").glob("version_*/metrics.csv"),
        key=lambda x: int(x.parent.stem.removeprefix("version_")),
    )
    processed_metrics_df = pd.concat(
        process_metrics_csv(p) for p in metrics_paths
    )
    processed_metrics_df.to_csv("metrics.csv", index=False)

    Path("JOB_FINISHED").touch()

    sys.exit(127)
