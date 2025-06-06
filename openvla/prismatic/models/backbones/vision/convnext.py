"""
convnext.py
"""

from prismatic.models.backbones.vision.base_vision import TimmViTBackbone, TimmConvnextBackbone

# Registry =>> Supported convnext Vision Backbones (from TIMM)
CONVNEXT_VISION_BACKBONES = {
    "convnext-xxlarge-clip-laion2b": "convnext_xxlarge.clip_laion2b_soup",
    "convnext-base-clip-laion2b": "convnext_base.clip_laion2b",
}


class ConvNextCLIPBackbone(TimmConvnextBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__(
            vision_backbone_id,
            CONVNEXT_VISION_BACKBONES[vision_backbone_id],
            image_resize_strategy,
            default_image_size=default_image_size,
        )
