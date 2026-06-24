import albumentations as A
import lightning as L
import numpy as np
import skimage as si
import torch as t
from joblib import cpu_count

from .utils import is_interactive, maybe_tqdm, maybe_trange


class SemanticSegmentationDataset(t.utils.data.Dataset):
    def __init__(
        self,
        image_arrs,
        mask_arrs=None,
        *,
        image_and_mask_transform=None,
        image_transform=None,
        mask_transform=None,
        mask_padding_val=None,
        standardize=True,
    ):
        self.image_arrs = image_arrs
        self.mask_arrs = mask_arrs
        self.image_and_mask_transform = (
            A.Compose(
                [image_and_mask_transform],
                additional_targets={"unpadded_region_mask": "mask"},
            )
            if image_and_mask_transform is not None
            else None
        )
        self.image_transform = image_transform
        self.mask_transform = (
            A.Compose(
                [mask_transform],
                additional_targets={"unpadded_region_mask": "mask"},
            )
            if mask_transform is not None
            else None
        )
        self.mask_padding_val = mask_padding_val
        self.standardize = standardize

    def __len__(self):
        return len(self.image_arrs)

    def __getitem__(self, idx):
        image_arr = self.image_arrs[idx]
        image_arr = si.util.img_as_float32(image_arr)
        h, w = image_arr.shape[:2]
        if self.mask_arrs is not None:
            mask_arr = self.mask_arrs[idx]
            mask_arr = si.util.img_as_float32(mask_arr)
        else:
            mask_arr = None
        unpadded_region_mask_arr = np.ones((h, w), dtype=np.float32)
        transformed_unpadded_region_mask_arr = unpadded_region_mask_arr
        transformed_image_arr = image_arr
        transformed_mask_arr = mask_arr
        if self.image_and_mask_transform is not None:
            transformed = self.image_and_mask_transform(
                unpadded_region_mask=transformed_unpadded_region_mask_arr,
                image=transformed_image_arr,
                mask=transformed_mask_arr,
            )
            transformed_unpadded_region_mask_arr = transformed[
                "unpadded_region_mask"
            ]
            transformed_image_arr = transformed["image"]
            transformed_mask_arr = transformed["mask"]
            if transformed_mask_arr is not None:
                if transformed_mask_arr is mask_arr:
                    transformed_mask_arr = transformed_mask_arr.copy()
                transformed_mask_arr[
                    transformed_unpadded_region_mask_arr < 0.5  # noqa: PLR2004
                ] = self.mask_padding_val
        if self.image_transform is not None:
            transformed_image_arr = self.image_transform(
                image=transformed_image_arr
            )["image"]
        if self.mask_transform is not None:
            transformed = self.mask_transform(
                unpadded_region_mask=transformed_unpadded_region_mask_arr,
                image=transformed_mask_arr,
            )
            transformed_unpadded_region_mask_arr = transformed[
                "unpadded_region_mask"
            ]
            transformed_mask_arr = transformed["image"]
            if transformed_mask_arr is not None:
                if transformed_mask_arr is mask_arr:
                    transformed_mask_arr = transformed_mask_arr.copy()
                transformed_mask_arr[
                    transformed_unpadded_region_mask_arr < 0.5  # noqa: PLR2004
                ] = self.mask_padding_val
        if self.standardize:
            if transformed_image_arr is image_arr:
                transformed_image_arr = transformed_image_arr.copy()
            transformed_image_arr -= transformed_image_arr.mean(
                axis=(0, 1),
                keepdims=True,
            )
            transformed_image_arr_stds = transformed_image_arr.std(
                axis=(0, 1),
                keepdims=True,
            )
            transformed_image_arr_stds[
                transformed_image_arr_stds < 1e-6  # noqa: PLR2004
            ] = 1e-6
            transformed_image_arr /= transformed_image_arr_stds
        to_return = [
            t.from_numpy(transformed_unpadded_region_mask_arr),
            t.from_numpy(transformed_image_arr).moveaxis(-1, 0),
        ]
        if transformed_mask_arr is not None:
            to_return.append(
                t.from_numpy(transformed_mask_arr).moveaxis(-1, 0)
            )
        return to_return


class SemanticSegmentationDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        train_image_arrs=None,
        train_mask_arrs=None,
        val_image_arrs=None,
        val_mask_arrs=None,
        train_image_and_mask_transform=None,
        train_image_transform=None,
        train_mask_transform=None,
        val_image_and_mask_transform=None,
        val_image_transform=None,
        val_mask_transform=None,
        mask_padding_val=None,
        standardize=True,
        batch_size,
    ):
        super().__init__()

        self.train_image_arrs = train_image_arrs
        self.train_mask_arrs = train_mask_arrs
        self.val_image_arrs = val_image_arrs
        self.val_mask_arrs = val_mask_arrs
        self.train_image_and_mask_transform = (
            train_image_and_mask_transform
        )
        self.train_image_transform = train_image_transform
        self.train_mask_transform = train_mask_transform
        self.val_image_and_mask_transform = val_image_and_mask_transform
        self.val_image_transform = val_image_transform
        self.val_mask_transform = val_mask_transform
        self.mask_padding_val = mask_padding_val
        self.standardize = standardize
        self.batch_size = batch_size

        self.predict_dataloader = self.val_dataloader

    def setup(self, stage):
        if (stage == "fit") and (self.train_image_arrs is not None):
            self.train_dataset = SemanticSegmentationDataset(
                self.train_image_arrs,
                self.train_mask_arrs,
                image_and_mask_transform=self.train_image_and_mask_transform,
                image_transform=self.train_image_transform,
                mask_transform=self.train_mask_transform,
                mask_padding_val=self.mask_padding_val,
                standardize=self.standardize,
            )

            train_class_counts = []

            for _ in maybe_trange(5, leave=False):
                for (
                    _,
                    _,
                    masks,
                ) in maybe_tqdm(self.train_dataloader(), leave=False):
                    train_class_counts.extend(masks.mean(axis=(2, 3)))
            train_class_counts = t.stack(train_class_counts)
            train_mean_class_counts = train_class_counts.mean(axis=0)
            self.train_class_freqs = (
                train_mean_class_counts
                / train_mean_class_counts.sum(dim=-1, keepdim=True)
            )
            if not t.all(self.train_class_freqs):
                msg = "self.train_class_freqs contains zeroes"
                raise ValueError(msg)

        if stage in {"fit", "predict"}:
            if self.val_image_arrs is not None:
                self.val_dataset = SemanticSegmentationDataset(
                    self.val_image_arrs,
                    self.val_mask_arrs,
                    image_and_mask_transform=self.val_image_and_mask_transform,
                    image_transform=self.val_image_transform,
                    mask_transform=self.val_mask_transform,
                    mask_padding_val=self.mask_padding_val,
                    standardize=self.standardize,
                )
        else:
            msg = f"{stage=}"
            raise NotImplementedError(msg)

    def train_dataloader(self):
        if self.trainer is not None:
            pin_memory = isinstance(
                self.trainer.accelerator,
                L.pytorch.accelerators.cuda.CUDAAccelerator,
            )
        else:
            pin_memory = t.cuda.is_available()
        return t.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
            if is_interactive()
            else cpu_count(only_physical_cores=True),
            pin_memory=pin_memory,
            persistent_workers=not is_interactive(),
        )

    def val_dataloader(self):
        if self.trainer is not None:
            pin_memory = isinstance(
                self.trainer.accelerator,
                L.pytorch.accelerators.cuda.CUDAAccelerator,
            )
        else:
            pin_memory = t.cuda.is_available()
        return t.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
            if is_interactive()
            else cpu_count(only_physical_cores=True),
            pin_memory=pin_memory,
            persistent_workers=not is_interactive(),
        )
