from sys import stdout, stderr
from os import listdir, mkdir
from os.path import exists, join
import argparse
import json
from datetime import timedelta
import pandas as pd
import cv2
import albumentations
import lightning as L
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from .utils import pd_merge_from
from .data import SemanticSegmentationDataModule, TASK
from .models import SEGFORMER_MODEL_VARIANT_CONFIG_OVERRIDES
from .modules import SemanticSegmentationModule
from .augmentations import ToolPasting, RandomLighting


if __name__ == "__main__":
    stdout.reconfigure(line_buffering=True)
    stderr.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir")
    parser.add_argument("data_dir")
    parser.add_argument("--rescale_target_height", type=int, default=128)
    parser.add_argument("--pad_to_shape", type=int, nargs=2, default=(128, 288))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--initial_lr", type=float, default=4 * 1e-4)
    parser.add_argument("--max_lr", type=float, default=1e-2)
    parser.add_argument("--min_lr", type=float, default=4 * 1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--segformer_model_variant", default="MiT-b0")
    parser.add_argument(
        "--segformer_attention_probs_dropout_prob", type=float, default=0
    )
    parser.add_argument("--segformer_classifier_dropout_prob", type=float, default=0.1)
    parser.add_argument("--segformer_drop_path_rate", type=float, default=0.1)
    parser.add_argument("--segformer_hidden_dropout_prob", type=float, default=0)
    parser.add_argument("--elastic_transform", action="store_true")
    parser.add_argument("--n_tool_augmentations", type=int, default=1)
    parser.add_argument("--black_tools", action="store_true")
    parser.add_argument("--lighting_augmentation", action="store_true")
    parser.add_argument("--sanity_check", action="store_true")
    args = parser.parse_args()

    args_json = json.dumps(vars(args))
    print(f"{args_json=}")

    if not exists(args.job_dir):
        mkdir(args.job_dir)
    else:
        print(f"{args.job_dir=} already exists and will be re-used")
        if ld_jd := listdir(args.job_dir):
            print(f"{args.job_dir=} is not empty:")
            print(ld_jd)

    with open(join(args.job_dir, "cl_args.json"), "w", encoding="utf-8") as out_file:
        out_file.write(args_json + "\n")

    csv_names = [
        "images_labels.csv",
        "images_videos.csv",
        "videos_subsets.csv",
    ]
    metadata_df = pd_merge_from(
        map(
            pd.read_csv,
            (join(args.data_dir, csv_name) for csv_name in csv_names),
        )
    )
    for col in metadata_df.columns:
        metadata_df[col] = metadata_df[col].apply(
            lambda x: (
                join(args.data_dir, x.removeprefix("data/"))
                if isinstance(x, str) and x.startswith("data/")
                else x
            )
        )
    if args.sanity_check:
        metadata_df = metadata_df.sample(frac=0.05)
    print(f"{len(metadata_df)=}")

    task = TASK
    print(f"{task=}")

    if args.n_tool_augmentations > 0:
        tools_df = pd_merge_from(
            map(
                pd.read_csv,
                (
                    join(args.data_dir, csv_name)
                    for csv_name in (
                        "tools_extraction_tips_videos.csv",
                        "videos_subsets.csv",
                    )
                ),
            )
        )
        for col in tools_df.columns:
            tools_df[col] = tools_df[col].apply(
                lambda x: (
                    join(args.data_dir, x.removeprefix("data/"))
                    if isinstance(x, str) and x.startswith("data/")
                    else x
                )
            )
        tools_df_train = tools_df[tools_df["subset"] == "train"]
        tools_df_valid = tools_df[tools_df["subset"] == "valid"]
        print(f"{len(tools_df_train)=}")
        print(f"{len(tools_df_valid)=}")

    # mask_value should always be 0.
    # Real mask padding happens inside __getitem__ method of
    # LapCholeDataset class, see HACK description at the top of that class
    train_geometric_augmentations = [
        albumentations.augmentations.geometric.Rotate(
            (-20, 20),
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.5,
        ),
        albumentations.core.composition.OneOf(
            [
                albumentations.augmentations.crops.transforms.RandomResizedCrop(
                    *args.pad_to_shape,
                    (0.8, 1.0),
                    (1.0, 1.0),
                    p=1,
                ),
                albumentations.augmentations.geometric.transforms.Affine(
                    scale=(0.8, 1.0),
                    keep_ratio=True,
                    cval=0,
                    cval_mask=0,
                    mode=cv2.BORDER_CONSTANT,
                    p=1,
                ),
            ],
            p=0.5,
        ),
    ]
    if args.elastic_transform:
        train_geometric_augmentations.append(
            albumentations.augmentations.geometric.ElasticTransform(
                sigma=10,
                alpha_affine=10,
                p=0.5,
            ),
        )
    train_geometric_augmentation = albumentations.core.composition.Sequential(
        train_geometric_augmentations, p=1
    )
    train_padding = albumentations.augmentations.geometric.PadIfNeeded(
        *args.pad_to_shape,
        position="random",
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        mask_value=0,
        always_apply=True,
    )
    train_geometric_transforms = [train_padding, train_geometric_augmentation]
    train_geometric_transform = albumentations.core.composition.Compose(
        train_geometric_transforms,
        additional_targets={"unpadded_region_mask": "mask"},
        p=1,
    )
    train_pixel_augmentations = [
        albumentations.augmentations.transforms.ColorJitter(hue=0.05, p=0.5),
        albumentations.augmentations.transforms.RandomFog(0.1, 0.8, 0.0, p=0.5),
        albumentations.augmentations.transforms.GaussNoise((0.001, 0.01), p=0.5),
    ]
    if args.lighting_augmentation:
        train_pixel_augmentations.insert(0, RandomLighting(p=1.0))
    if args.n_tool_augmentations > 0:
        if args.black_tools:
            tool_pasting_aug = ToolPasting(tools_df_train, fill_color=0, p=1.0)
        else:
            tool_pasting_aug = ToolPasting(tools_df_train, p=1.0)
        for i in range(args.n_tool_augmentations):
            train_pixel_augmentations.insert(0, tool_pasting_aug)
    train_pixel_transform = albumentations.core.composition.Sequential(
        train_pixel_augmentations, p=1
    )
    # SegFormer models output downscaled predictions.
    # NOTE: as clarified in a NOTE in SemanticSegmentationDataModule's
    # __getitem__ method implementation, mask transforms are implemented as
    # image transforms from albumentations' point of view.
    mask_downscaling_transform = albumentations.core.composition.Compose(
        [
            albumentations.augmentations.geometric.resize.Resize(
                args.pad_to_shape[0] // 4,
                args.pad_to_shape[1] // 4,
                always_apply=True,
            )
        ],
        additional_targets={"unpadded_region_mask": "mask"},
        p=1,
    )
    val_geometric_transform = albumentations.core.composition.Compose(
        [
            albumentations.augmentations.geometric.PadIfNeeded(
                *args.pad_to_shape,
                position="center",
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                always_apply=True,
            )
        ],
        additional_targets={"unpadded_region_mask": "mask"},
        p=1,
    )
    val_pixel_transform = None

    standardize = True

    dm = SemanticSegmentationDataModule(
        metadata_df,
        args.batch_size,
        args.batch_size,
        args.rescale_target_height,
        train_geometric_transform,
        train_pixel_transform,
        mask_downscaling_transform,
        val_geometric_transform,
        val_pixel_transform,
        mask_downscaling_transform,
        standardize,
    )
    num_labels = len(task.class_names)
    print(f"{num_labels=}")

    model_class = SegformerForSemanticSegmentation
    if args.sanity_check and (args.segformer_model_variant != "sanity_check"):
        print(
            f'{args.sanity_check=}, but {args.segformer_model_variant=}. Changing it to "sanity_check"'
        )
        args.segformer_model_variant = "sanity_check"
    model_config_override = SEGFORMER_MODEL_VARIANT_CONFIG_OVERRIDES[
        args.segformer_model_variant
    ] | {
        "num_labels": num_labels,
        "attention_probs_dropout_prob": args.segformer_attention_probs_dropout_prob,
        "classifier_dropout_prob": args.segformer_classifier_dropout_prob,
        "drop_path_rate": args.segformer_drop_path_rate,
        "hidden_dropout_prob": args.segformer_hidden_dropout_prob,
    }
    model_config = SegformerConfig(**model_config_override)
    model_args = [model_config]

    module = SemanticSegmentationModule(
        model_class,
        model_args,
        {},
        args.initial_lr,
        args.max_lr,
        args.min_lr,
        args.weight_decay,
    )

    checkpoint_callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=args.job_dir,
            filename="{epoch:03d}_{val_loss:.5f}",
            monitor="val_loss",
            mode="min",
        ),
    ]

    timer_callback = L.pytorch.callbacks.Timer()

    logger = L.pytorch.loggers.CSVLogger(args.job_dir, flush_logs_every_n_steps=1)

    if args.sanity_check and (args.epochs > 2):
        print(f"{args.sanity_check=}, but {args.epochs=}. Changing it to 2")
        args.epochs = 2
    trainer = L.Trainer(
        logger=logger,
        callbacks=[*checkpoint_callbacks, timer_callback],
        max_epochs=args.epochs,
        log_every_n_steps=1,
        enable_progress_bar=False,
        default_root_dir=args.job_dir,
    )
    trainer.fit(module, datamodule=dm)
    print("Time elapsed:")
    for k, v in timer_callback.state_dict()["time_elapsed"].items():
        if k in ("train", "validate"):
            print(f"{k}: {timedelta(seconds=v)}")
    print("Best checkpoint paths:")
    for c_c in checkpoint_callbacks:
        print(c_c.best_model_path)
    metrics_path = logger.experiment.metrics_file_path
    metrics_df = (
        pd.read_csv(metrics_path)
        .sort_values(by=["epoch", "step"])
        .drop(columns=["step"])
    )
    metrics_df = metrics_df[sorted(metrics_df.columns)]
    metrics_df_grouped = metrics_df.groupby("epoch")
    for epoch, epoch_df in metrics_df_grouped:
        if len(epoch_df) > 2:
            raise RuntimeError(
                f"In metrics DataFrame there are {len(epoch_df)=} > 2 rows for {epoch=}"
            )
    metrics_df = metrics_df_grouped.first().sort_index()
    metrics_df.to_csv(metrics_path.removesuffix(".csv") + "_processed.csv")
