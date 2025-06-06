# coding=utf-8
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Image processor class for Magma."""

from typing import List, Optional, Union
import ast
import numpy as np
import torchvision
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import (
    convert_to_rgb,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
    make_list_of_images,
    valid_images,
)
from transformers.utils import TensorType, is_vision_available, logging

from transformers import AutoImageProcessor

logger = logging.get_logger(__name__)


if is_vision_available():
    from PIL import Image

import torch
import torchvision

def padding_336(b):
    width, height = b.size
    tar = int(np.ceil(height / 336) * 336)
    top_padding = int((tar - height)/2)
    bottom_padding = tar - height - top_padding
    left_padding = 0
    right_padding = 0
    b = torchvision.transforms.functional.pad(b, [left_padding, top_padding, right_padding, bottom_padding], fill=[255,255,255])

    return b

def calc_padded_size(width, height, padding_unit=336):  
    target_height = int(np.ceil(height / padding_unit) * padding_unit)  
    top_padding = int((target_height - height) / 2)  
    bottom_padding = target_height - height - top_padding  
    left_padding = 0  
    right_padding = 0  
    padded_width = width + left_padding + right_padding  
    padded_height = height + top_padding + bottom_padding  
    return padded_width, padded_height  

def HD_transform(img, hd_num=4, base_img_size=768):
    width, height = img.size
    trans = False
    if width < height:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = (width / height)
    scale = 1
    while scale*np.ceil(scale/ratio) <= hd_num:
        scale += 1
    scale -= 1
    new_w = int(scale * base_img_size)
    new_h = int(new_w / ratio)

    img = torchvision.transforms.functional.resize(img, [new_h, new_w],)
    img = padding_336(img)
    width, height = img.size
    if trans:
        img = img.transpose(Image.TRANSPOSE)

    return img

def calc_hd_transform_size(width, height, hd_num=16):  
    transposed = False  
    if width < height:  
        width, height = height, width  
        transposed = True  
  
    ratio = width / height  
    scale = 1  
    while scale * np.ceil(scale / ratio) <= hd_num:  
        scale += 1  
    scale -= 1  
  
    new_width = int(scale * 336)  
    new_height = int(new_width / ratio)  
  
    padded_width, padded_height = calc_padded_size(new_width, new_height)  
      
    if transposed:  
        padded_width, padded_height = padded_height, padded_width  
  
    return padded_width, padded_height  

def pad_to_max_num_crops_tensor(images, max_crops=5):
    """
    images: B x 3 x H x W, B<=max_crops
    """
    B, _, H, W = images.shape
    if B < max_crops:
        pad = torch.zeros(max_crops - B, 3, H, W, dtype=images.dtype, device=images.device)
        images = torch.cat([images, pad], dim=0)
    return images

def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit

def process_anyres_image(image, max_num_crops=None, base_width=768, base_height=768):
    """
    Process an image with variable resolutions.

    Args:
        image (torch.Tensor): The input image to be processed.
        max_num_crops (int): Maximum number of crops 

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    assert max_num_crops is not None
    grid_pinpoints = []
    for i in range(1, max_num_crops+1):
        for j in range(1, max_num_crops // i + 1):
            grid_pinpoints.append((i, j))
    grid_pinpoints = [(int(res[0] * base_width), int(res[1] * base_height)) for res in grid_pinpoints]

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    
    best_resolution = select_best_resolution((image.shape[2], image.shape[1]), possible_resolutions)
    # NOTE: reverse best_resolution from (width, height) to (height, width)
    best_resolution = (best_resolution[1], best_resolution[0])
    best_resolution_grid = (best_resolution[0] // base_height, best_resolution[1] // base_width)

    # resize image tensor to best resolution
    image = torch.nn.functional.interpolate(image[None,:,:,:], size=best_resolution, mode='bilinear')
    # divide image tensor into patches
    patches = image.unfold(2, base_height, base_height).unfold(3, base_width, base_width)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(best_resolution_grid[0]*best_resolution_grid[1], -1, base_height, base_width)
    return (patches, best_resolution_grid)

def process_anyres_image_global(image, max_num_crops=None, base_width=768, base_height=768):
    """
    Process an image with variable resolutions.

    Args:
        image (torch.Tensor): The input image to be processed.
        max_num_crops (int): Maximum number of crops 

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    assert max_num_crops is not None
    grid_pinpoints = []
    for i in range(1, max_num_crops+1):
        for j in range(1, max_num_crops // i + 1):
            grid_pinpoints.append((i, j))
    grid_pinpoints = [(int(res[0] * base_width), int(res[1] * base_height)) for res in grid_pinpoints]

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    
    best_resolution = select_best_resolution((image.shape[2], image.shape[1]), possible_resolutions)
    # NOTE: reverse best_resolution from (width, height) to (height, width)
    best_resolution = (best_resolution[1], best_resolution[0])
    best_resolution_grid = (best_resolution[0] // base_height, best_resolution[1] // base_width)

    # resize image tensor to best resolution
    image = torch.nn.functional.interpolate(image[None,:,:,:], size=best_resolution, mode='bilinear')
    return image

class preprocessor():
    def __init__(self, image_preprocessor, base_resolution=(256, 256)):
        self.image_preprocessor = image_preprocessor
        self.crop_size = {
            'height': base_resolution[0],
            'width': base_resolution[1]
        }
        self.image_mean = image_preprocessor.transforms[-1].mean

    def preprocess(self, image, return_tensors='pt'):
        image = self.image_preprocessor(image).unsqueeze(0)      
        return {
            'pixel_values': image,
        }   

class MagmaImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Magma image processor. Based on [`CLIPImageProcessor`] with incorporation of additional techniques
    for processing high resolution images as explained in the [InternLM-XComposer2-4KHD](https://arxiv.org/pdf/2404.06512)

    Args:
        anyres_strategy (`str`):
            strategy to cope with high-resolution images. one conventional way is multi-crop and many other works to accomadate clip-vit models. 
            however, since we are using convnext, which is essentially convnet, so we can use arbitary resolution images. as such, we use global strategy by defualt,
            i.e., directly resize image holistically to a certain resolution.
        base_img_size (int, *optional*, defaults to 768):
            as convnext has 1/32 downsample rate, we use 768 as the base resolution so that the resulted feature map is 24x24.
        num_crops (int, *optional*, defaults to 1):
            number of effective crops when coping with images with higher resolution than 768x768. note that num_crops > 1 does not mean we are cropping the image.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        anyres_strategy: str = 'global', 
        base_img_size: int = 768, 
        num_crops: int = 1,
        do_convert_rgb: bool = True,
        image_mean: List[float] = OPENAI_CLIP_MEAN,
        image_std: List[float] = OPENAI_CLIP_STD,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.base_img_size = base_img_size
        self.anyres_strategy = anyres_strategy
        self.num_crops = num_crops
        self.do_convert_rgb = do_convert_rgb
        self.image_mean = image_mean
        self.image_std = image_std

    def preprocess(
        self,
        images: Union[ImageInput, List[ImageInput]],
        do_pad: bool = False,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):
        """
        Args:
            images (`ImageInput` or `List[ImageInput]`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
        """
        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
                
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]
  
        # tensor transform and normalize
        img_processor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.image_mean, self.image_std)
        ])      

        images = [img_processor(image) for image in images]
        image_data_type = 'half' if images[0].type() == 'torch.HalfTensor' else 'float'
        images = [image.float() for image in images]

        # crop images to the same size
        image_patches = [process_anyres_image(image, self.num_crops, base_width=self.base_img_size, base_height=self.base_img_size) for image in images]
        pixel_values = torch.cat([image[0] for image in image_patches], dim=0)
        # pixel_values = [image[0] for image in image_patches]
        image_sizes = [image_patch[1] for image_patch in image_patches]     

        if image_data_type == 'half':
            pixel_values = pixel_values.half()

        data = {
            "pixel_values": pixel_values, 
            "image_sizes": image_sizes,
        }
        return BatchFeature(data=data, tensor_type=return_tensors)

AutoImageProcessor.register("MagmaImageProcessor", MagmaImageProcessor)