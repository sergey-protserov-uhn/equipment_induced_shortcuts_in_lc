from types import MethodType
from functools import cache, cached_property
from typing import Any
from dataclasses import dataclass
import numpy as np
import skimage as si
import torch as t
from torch.utils.data import DataLoader
import lightning as L
from .preprocess import process_mask_arr
from .utils import rescale_to_height, image_arr2brightest_square


@dataclass
class Task:
    name: Any
    class_names: Any
    mask_path_col_name: Any
    padding_val: Any


# WARN: background is assumed to always be the last class
TASK = Task(
    "LC_dangerous_safe",
    ["dangerous", "safe", "bg"],
    "label_path",
    [0.0, 0.0, 1.0],
)


# HACK: all of the augmentation libraries that we tried have grave
# limitations in terms of padding with multi-channel values.
# Therefore, "true" padding will happen in __getitem__ method of this
# class, whereas external transforms will pad an array of ones for us
# to show which region corresponds to original input and where this input
# was padded. Having such a mask is useful for validation and inference anyway.
class SemanticSegmentationDataset(t.utils.data.Dataset):
    def __init__(
        self,
        metadata_df,
        rescale_target_height,
        image_and_mask_transform,
        image_transform,
        mask_transform,
        standardize,
    ):
        self.metadata_df = metadata_df
        self.rescale_target_height = rescale_target_height
        self.image_and_mask_transform = image_and_mask_transform
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.standardize = standardize

    def __len__(self):
        return len(self.metadata_df)

    @cache
    def prepare_item(self, idx):
        metadata_row = self.metadata_df.iloc[idx]
        image_arr = si.io.imread(metadata_row["image_path"])[..., :3]
        image_arr = rescale_to_height(image_arr, self.rescale_target_height, 1)
        image_arr = si.util.img_as_float32(image_arr)
        unpadded_region_mask_arr = np.ones(image_arr.shape[:2], dtype=np.float32)
        mask_arr = si.io.imread(metadata_row[TASK.mask_path_col_name])
        mask_arr = process_mask_arr(mask_arr, image_arr.shape[:2])
        if (mask_chans := mask_arr.shape[2]) != len(TASK.class_names):
            raise RuntimeError(
                f"For {TASK=}, {mask_chans=} shows wrong amount of channels"
            )
        mask_arr = si.util.img_as_float32(mask_arr)
        return unpadded_region_mask_arr, image_arr, mask_arr

    def __getitem__(self, idx):
        (
            unpadded_region_mask_arr,
            image_arr,
            mask_arr,
        ) = self.prepare_item(idx)
        transformed_unpadded_region_mask_arr = unpadded_region_mask_arr
        transformed_image_arr = image_arr
        transformed_mask_arr = mask_arr
        if self.image_and_mask_transform is not None:
            transformed = self.image_and_mask_transform(
                unpadded_region_mask=transformed_unpadded_region_mask_arr,
                image=transformed_image_arr,
                mask=transformed_mask_arr,
            )
            # transformed_unpadded_region_mask_arr from the line below will be used
            # for padding and also returned from __getitem__, useful for ignoring
            # padded regions during validation, or undoing padding during inference.

            # There is no guarantee wrt whether transforms return
            # copies or original objects, therefore we perform copies
            # to have independent objects, guaranteed
            transformed_unpadded_region_mask_arr = transformed["unpadded_region_mask"]
            if transformed_unpadded_region_mask_arr is unpadded_region_mask_arr:
                transformed_unpadded_region_mask_arr = (
                    transformed_unpadded_region_mask_arr.copy()
                )
            transformed_unpadded_region_mask_arr = (
                transformed_unpadded_region_mask_arr.round()
            )
            transformed_image_arr = transformed["image"]
            if transformed_image_arr is image_arr:
                transformed_image_arr = transformed_image_arr.copy()
            transformed_mask_arr = transformed["mask"]
            if transformed_mask_arr is mask_arr:
                transformed_mask_arr = transformed_mask_arr.copy()
            # See HACK description above this class' definition
            transformed_mask_arr[transformed_unpadded_region_mask_arr < 0.5] = (
                TASK.padding_val
            )
        if self.image_transform is not None:
            transformed_image_arr = self.image_transform(image=transformed_image_arr)[
                "image"
            ]
            if transformed_image_arr is image_arr:
                transformed_image_arr = transformed_image_arr.copy()
        if self.mask_transform is not None:
            transformed = self.mask_transform(
                unpadded_region_mask=transformed_unpadded_region_mask_arr,
                # NOTE: the augmentation library that we use, Albumentations,
                # has ImageOnlyTransform interface, but nothing like
                # MaskOnlyTransform. ImageOnlyTransform, however, doesn't
                # do anything image-specific, and can be applied to masks.
                # Therefore, we implement mask transforms as "image-only",
                # and pass a mask in place of an image.
                # This is not a hack, just a naming quirk.
                image=transformed_mask_arr,
            )
            transformed_unpadded_region_mask_arr = transformed["unpadded_region_mask"]
            if transformed_unpadded_region_mask_arr is unpadded_region_mask_arr:
                transformed_unpadded_region_mask_arr = (
                    transformed_unpadded_region_mask_arr.copy()
                )
            transformed_unpadded_region_mask_arr = (
                transformed_unpadded_region_mask_arr.round()
            )
            transformed_mask_arr = transformed["image"]
            if transformed_mask_arr is mask_arr:
                transformed_mask_arr = transformed_mask_arr.copy()
            # See HACK description above this class' definition
            transformed_mask_arr[transformed_unpadded_region_mask_arr < 0.5] = (
                TASK.padding_val
            )
        if self.standardize:
            _, brightest_square = image_arr2brightest_square(
                transformed_image_arr, size_frac=0.8, num_cands=(8, 18)
            )
            transformed_image_arr -= brightest_square.mean(axis=(0, 1), keepdims=True)
            transformed_image_arr /= brightest_square.std(axis=(0, 1), keepdims=True)
        return (
            t.from_numpy(transformed_unpadded_region_mask_arr),
            t.from_numpy(transformed_image_arr).moveaxis(-1, 0),
            t.from_numpy(transformed_mask_arr).moveaxis(-1, 0),
        )

    @cached_property
    def class_frequencies(self):
        class_counts = []
        for i in range(10):
            for *_, transformed_mask_ten in self:
                # WARN: assuming 2D masks
                class_counts.append(transformed_mask_ten.sum(axis=(1, 2)))
        mean_class_counts = t.stack(class_counts).mean(axis=0)
        class_freqs = mean_class_counts / mean_class_counts.sum()
        return class_freqs


class SemanticSegmentationDataModule(L.LightningDataModule):
    def __init__(
        self,
        metadata_df,
        train_batch_size,
        val_batch_size,
        rescale_target_height,
        train_image_and_mask_transform,
        train_image_transform,
        train_mask_transform,
        val_image_and_mask_transform,
        val_image_transform,
        val_mask_transform,
        standardize,
    ):
        super().__init__()
        self.metadata_df = metadata_df
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.rescale_target_height = rescale_target_height
        self.train_image_and_mask_transform = train_image_and_mask_transform
        self.train_image_transform = train_image_transform
        self.train_mask_transform = train_mask_transform
        self.val_image_and_mask_transform = val_image_and_mask_transform
        self.val_image_transform = val_image_transform
        self.val_mask_transform = val_mask_transform
        self.standardize = standardize

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = SemanticSegmentationDataset(
                self.metadata_df[self.metadata_df["subset"] == "train"],
                self.rescale_target_height,
                self.train_image_and_mask_transform,
                self.train_image_transform,
                self.train_mask_transform,
                self.standardize,
            )
            self.val_dataset = SemanticSegmentationDataset(
                self.metadata_df[self.metadata_df["subset"] == "valid"],
                self.rescale_target_height,
                self.val_image_and_mask_transform,
                self.val_image_transform,
                self.val_mask_transform,
                self.standardize,
            )
            # WARN: this determinism is only across different validation epochs
            # within a single job run. Different jobs will still have different
            # validation transform sequences
            self.val_dataset.__getitem__ = cache(
                MethodType(type(self.val_dataset).__getitem__, self.val_dataset)
            )
            print(f"{len(self.train_dataset)=}")
            print(f"{len(self.val_dataset)=}")
            print(f"{len(self.train_dataloader())=}")
            print(f"{len(self.val_dataloader())=}")
        else:
            raise NotImplementedError(f"Code for {stage=} is not implemented")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            pin_memory=True,
        )
