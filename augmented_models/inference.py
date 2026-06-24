import random
import numpy as np
import skimage as si
import cv2
import albumentations
import torch as t
from torch import nn
from .utils import rescale_to_height, image_arr2brightest_square


def image_arr2image_unpadded_region_mask_tens(
    image_arr,
    rescale_target_height=128,
    pad_to_shape=(128, 288),
    brightest_square_size_frac=0.8,
    brightest_square_num_cands=(8, 18),
):
    image_arr = image_arr[..., :3]
    image_arr = rescale_to_height(image_arr, rescale_target_height, 1)
    image_arr = si.util.img_as_float32(image_arr)
    orig_shape = image_arr.shape[:2]
    unpadded_region_mask_arr = np.ones(orig_shape, dtype=np.float32)
    padding = albumentations.core.composition.Compose(
        [
            albumentations.augmentations.geometric.PadIfNeeded(
                *pad_to_shape,
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
    padded = padding(image=image_arr, unpadded_region_mask=unpadded_region_mask_arr)
    image_arr = padded["image"]
    unpadded_region_mask_arr = padded["unpadded_region_mask"]
    _, brightest_square = image_arr2brightest_square(
        image_arr,
        size_frac=brightest_square_size_frac,
        num_cands=brightest_square_num_cands,
    )
    image_arr -= brightest_square.mean(axis=(0, 1), keepdims=True)
    image_arr /= brightest_square.std(axis=(0, 1), keepdims=True)
    return (
        t.from_numpy(unpadded_region_mask_arr),
        orig_shape,
        t.from_numpy(image_arr).moveaxis(-1, 0),
    )


@t.inference_mode()
def image_arrs2probss(
    image_arrs,
    module,
    device,
    rescale_target_height=128,
    pad_to_shape=(128, 288),
    brightest_square_size_frac=0.8,
    brightest_square_num_cands=(8, 18),
):
    module.eval()
    module.to(device)
    unpadded_region_mask_tens = []
    orig_shapes = []
    image_tens = []
    for image_arr in image_arrs:
        unpadded_region_mask_ten, orig_shape, image_ten = (
            image_arr2image_unpadded_region_mask_tens(
                image_arr,
                rescale_target_height=rescale_target_height,
                pad_to_shape=pad_to_shape,
                brightest_square_size_frac=brightest_square_size_frac,
                brightest_square_num_cands=brightest_square_num_cands,
            )
        )
        unpadded_region_mask_tens.append(unpadded_region_mask_ten)
        orig_shapes.append(orig_shape)
        image_tens.append(image_ten)
    images_ten = t.stack(image_tens).to(device)
    output = module.model.forward(images_ten)
    probss = nn.functional.softmax(output.logits, dim=1)
    probss = nn.functional.interpolate(probss, scale_factor=4, mode="bilinear")
    probss = probss.moveaxis(1, -1).cpu()
    probss = [
        probs[unpadded_region_mask_ten > 0.5].reshape(*orig_shape, probss.shape[-1])
        for probs, unpadded_region_mask_ten, orig_shape in zip(
            probss, unpadded_region_mask_tens, orig_shapes
        )
    ]
    return probss


def image_arr2TTA_probs(image_arr, module, augs, n=100):
    image_arrs = [image_arr]
    for i in range(n):
        aug = random.choice(augs)
        image_arrs.append(aug(image=image_arr)["image"])
    return image_arrs2probss(image_arrs, module)
