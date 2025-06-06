from PIL import Image
from io import BytesIO
import base64
import random
import torch
import math
import ast
from transformers import StoppingCriteria
from .constants import IMAGE_TOKEN_INDEX


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        # sample a random between 0 and (width - height) // 2
        y_start = random.randint((width - height) // 2, (width - height) // 2 + 1)
        result.paste(pil_img, (0, y_start))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        # sample a random between 0 and (height - width) // 2
        x_start = random.randint((height - width) // 2, (height - width) // 2 + 1)
        result.paste(pil_img, (x_start, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    for image in images:
        if image_aspect_ratio == 'pad':
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
        image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        new_images.append(image)
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

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

def process_anyres_image(image, max_num_crops=None, base_width=384, base_height=384):
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

def process_anyres_image_global(image, max_num_crops=None, base_width=384, base_height=384):
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

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
