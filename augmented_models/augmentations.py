from functools import cache
import random
import numpy as np
import skimage as si
import cv2
import albumentations
from .utils import rescale_to_height


@cache
def si_io_imread(p):
    return si.io.imread(p)


class ToolPasting(albumentations.core.transforms_interface.ImageOnlyTransform):
    def __init__(self, tools_df, fill_color=None, always_apply=None, p=0.5):
        super().__init__(always_apply, p)
        self.tools_df = tools_df
        self.fill_color = fill_color

    @staticmethod
    @cache
    def tool_path2rescaled_tool_arr(tool_path, rescaling_factor):
        tool_arr = si_io_imread(tool_path)
        target_height = int(tool_arr.shape[0] * rescaling_factor)
        if target_height % 2 != 0:
            target_height += 1
        tool_arr_RGB = si.util.img_as_float32(
            rescale_to_height(tool_arr[..., :3], target_height, 1)
        )
        tool_arr_alpha = si.util.img_as_float32(
            rescale_to_height(tool_arr[..., 3], target_height, 0)
        )
        return np.concatenate((tool_arr_RGB, tool_arr_alpha[..., None]), axis=2)

    def apply(self, img, **params):
        tool_row = self.tools_df.sample()
        rescaling_factor = img.shape[0] / tool_row["original_image_height"].item()
        tool_arr = self.tool_path2rescaled_tool_arr(
            tool_row["tool_image_path"].item(), rescaling_factor
        )
        if tool_arr.dtype != img.dtype:
            raise RuntimeError(f"{tool_arr.dtype=} != {img.dtype=}")
        padding = albumentations.augmentations.geometric.PadIfNeeded(
            max(img.shape[0], tool_arr.shape[0]),
            max(img.shape[1], tool_arr.shape[1]),
            position="random",
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            always_apply=True,
        )
        tool_arr = padding(image=tool_arr)["image"]
        rotation = albumentations.augmentations.geometric.Rotate(
            180,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            always_apply=True,
        )
        while True:
            tool_arr_ = rotation(image=tool_arr)["image"]
            start_idx_0 = random.randint(0, tool_arr_.shape[0] - img.shape[0])
            start_idx_1 = random.randint(0, tool_arr_.shape[1] - img.shape[1])
            tool_arr_ = tool_arr_[
                start_idx_0 : start_idx_0 + img.shape[0],
                start_idx_1 : start_idx_1 + img.shape[1],
            ]
            if tool_arr_[..., 3].sum() > tool_arr[..., 3].sum() / 2:
                tool_arr = tool_arr_
                break
        tool_alpha_ge_05 = tool_arr[..., 3] > 0.5
        img = img.copy()
        if self.fill_color is None:
            img[tool_alpha_ge_05] = tool_arr[tool_alpha_ge_05, :3]
        else:
            img[tool_alpha_ge_05] = self.fill_color
        return img


def random_lighting(
    image_arr,
    n=1,
    range_range=(0.01, 10),
    loc_range=1.2,
    magnitude=50,
    return_blob_center_coord=False,
):
    image_arr_lab_ = si.color.rgb2lab(image_arr)
    h, w = image_arr.shape[:2]
    results = []
    for i in range(n):
        image_arr_lab = image_arr_lab_.copy()
        range_ = np.random.uniform(*range_range)
        loc_x = range_ * np.random.uniform(-loc_range, loc_range)
        loc_y = range_ * np.random.uniform(-loc_range, loc_range) * h / w
        x = np.linspace(-range_, range_, w)[None, :]
        y = np.linspace(-range_ * h / w, range_ * h / w, h)[:, None]
        gaussian_blob = np.exp(-0.5 * (x - loc_x) ** 2) * np.exp(
            -0.5 * (y - loc_y) ** 2
        )
        gaussian_blob -= gaussian_blob.min()
        gaussian_blob /= gaussian_blob.max()
        gaussian_blob *= magnitude
        image_arr_lab[..., 0] = np.clip(image_arr_lab[..., 0] + gaussian_blob, 0, 100)
        image_arr_ = si.color.lab2rgb(image_arr_lab)
        if not return_blob_center_coord:
            results.append(image_arr_)
        else:
            results.append(
                (
                    image_arr_,
                    np.unravel_index(gaussian_blob.argmax(), gaussian_blob.shape),
                )
            )
    return results


class RandomLighting(albumentations.core.transforms_interface.ImageOnlyTransform):
    def __init__(
        self,
        range_range=(0.01, 10),
        loc_range=1.2,
        magnitude=50,
        always_apply=None,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.range_range = range_range
        self.loc_range = loc_range
        self.magnitude = magnitude

    def apply(self, img, **params):
        return si.util.img_as_float32(
            random_lighting(
                img,
                n=1,
                range_range=self.range_range,
                loc_range=self.loc_range,
                magnitude=self.magnitude,
            )[0]
        )
