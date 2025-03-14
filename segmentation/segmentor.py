# From https://github.com/facebookresearch/dinov2/blob/main/notebooks/semantic_segmentation.ipynb

import math
import itertools
import urllib
from functools import partial
from PIL import Image
import numpy as np
import cv2
from typing import Union
import torch.utils.data
from pathlib import Path

import mmcv
from mmcv.runner import load_checkpoint

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from mmseg.apis import init_segmentor, inference_segmentor

DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


# The next line is necessary to register the model in mmsegmentation
import dinov2.eval.segmentation.models
import dinov2.eval.segmentation.utils.colormaps as COLORMAPS

class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


class Segmentor():

    cfg = None
    backbone_size = None
    head_type = None
    head_dataset = None
    head_scale_count = None
    backbone_model = None
    backbone_name = None
    model = None

    def __init__(self, 
                 backbone_size = "small",  # in ("small", "base", "large" or "giant")
                 head_type = "ms",  # in ("ms", "linear") 
                 head_dataset = "voc2012",  # in ("ade20k", "voc2012")
                 head_scale_count = 3  # in (1,2,3,4,5)
                 ):
        self.backbone_size = backbone_size
        self.head_type = head_type
        self.head_dataset = head_dataset
        self.head_scale_count = head_scale_count
        self.backbone_name = self._compute_backbone_name()

        self.cfg = self._get_head_config()
        self.backbone_model = self._load_backbone()
        self.model = self._create_segmentor()
        self._load_checkpoint()

    def get_segmentation_for_video(
        self, 
        video_path: Union[str, Path], 
        output_path: Union[str, Path],
        scale_factor: float = 1,
        fps: int = None,
        codec: str = 'mp4v',
        classes_only: str = None,
    ) -> None:
        """Process a video to generate depth estimation.
        
        Args:
            video_path: Path to the input video
            output_path: Path where to save the depth video
            scale_factor: Scale factor to apply to the images
            fps: Frames per second for the output video. If None, uses the input video fps
            codec: Video codec to use ('avc1', 'h264' or 'mp4v', default: 'mp4v')
            classes_only: Comma-separated list of classes to include in the output video. Only classes in this list will be rendered.
        """

        # Get the colormap
        colormap = self._get_colormap(classes_only)
        
        # Open the video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Configure video output
        output_fps = fps if fps is not None else original_fps
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(
            str(output_path), 
            fourcc, 
            output_fps,
            (new_width, new_height),
            isColor=True
        )

        if not out.isOpened():
            raise ValueError(f"Could not open video: {output_path} with codec: {codec}")

        try:
            for _ in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR frame to RGB and create PIL image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize if necessary
                if scale_factor != 1:
                    pil_image = Image.fromarray(frame_rgb)
                    pil_image = pil_image.resize((new_width, new_height))
                    img_array = np.array(pil_image)
                else:
                    img_array = frame_rgb
                
                # Process the frame using inference_segmentor
                result = inference_segmentor(self.model, img_array)[0]
                depth_image = self._render_segmentation(result.squeeze(), colormap)
                
                # Convert depth image to BGR for OpenCV
                depth_frame = cv2.cvtColor(np.array(depth_image), cv2.COLOR_RGB2BGR)
                out.write(depth_frame)
        
        finally:
            # Release resources
            cap.release()
            out.release()

    def get_segmentation_for_image(
        self,
        image: Image.Image,
        scale_factor: float = 1,
        classes_only: str = None,
    ) -> Image.Image:
        """Generate depth map for an image.
        
        Args:
            image: Input PIL image
            scale_factor: Scale factor to apply to the image resolution
            classes_only: Comma-separated list of classes to include in the output video. Only classes in this list will be rendered.
            
        Returns:
            PIL Image containing the depth map
        """
        # Get the colormap
        colormap = self._get_colormap(classes_only)

        if (scale_factor != 1):
            rescaled_image = image.resize((
                int(scale_factor * image.width), 
                int(scale_factor * image.height)
            ))
        else:
            rescaled_image = image

        # Convert PIL image to numpy array
        img_array = np.array(rescaled_image)
        result = inference_segmentor(self.model, img_array)[0]
        segmentation_image = self._render_segmentation(result.squeeze(), colormap)
        
        return segmentation_image

    def _render_segmentation(self, logits, colormap) -> Image:
        values = colormap[logits]
        return Image.fromarray(values)

    def _load_config_from_url(self, url: str) -> str:
        print(f"Loading config from {url}")
        with urllib.request.urlopen(url) as f:
            return f.read().decode()
    
    def _get_head_config(self):
        head_config_url = f"{DINOV2_BASE_URL}/{self.backbone_name}/{self.backbone_name}_{self.head_dataset}_{self.head_type}_config.py"

        cfg_str = self._load_config_from_url(head_config_url)
        cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

        if self.head_type == "ms":
            cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1]["img_ratios"][:self.head_scale_count]
            print("scales:", cfg.data.test.pipeline[1]["img_ratios"])

        return cfg
    
    def _compute_backbone_name(self):
        archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        arch = archs[self.backbone_size]
        name = f"dinov2_{arch}"
        return name
    
    def _get_colormap(self, classes_only: str = None):
        colormaps = {
            "ade20k": COLORMAPS.ADE20K_COLORMAP,
            "voc2012": COLORMAPS.VOC2012_COLORMAP,
        }
        classes = {
            "ade20k": COLORMAPS.ADE20K_CLASS_NAMES,
            "voc2012": COLORMAPS.VOC2012_CLASS_NAMES,
        }
        base = colormaps[self.head_dataset]

        if classes_only is not None:
            # Get the indices of the classes in the colormap
            classes_list = classes[self.head_dataset]
            classes_only_list = [c.strip() for c in classes_only.split(",")]
            excluded_indices = [i for i, c in enumerate(classes_list) if c not in classes_only_list]
            cloned_colormap = base.copy()
            for i in excluded_indices:
                cloned_colormap[i] = (0, 0, 0)
            base = cloned_colormap

        return np.array(base, dtype=np.uint8)
    
    def _load_backbone(self):
        model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=self.backbone_name)
        model.eval()
        model.cuda()

        return model
    
    def _load_checkpoint(self):
        head_checkpoint_url = f"{DINOV2_BASE_URL}/{self.backbone_name}/{self.backbone_name}_{self.head_dataset}_{self.head_type}_head.pth"
        load_checkpoint(self.model, head_checkpoint_url, map_location="cpu")
        self.model.eval()
        self.model.cuda()

    def _create_segmentor(self):
        model = init_segmentor(self.cfg)
        
        model.backbone.forward = partial(
            self.backbone_model.get_intermediate_layers,
            n=self.cfg.model.backbone.out_indices,
            reshape=True,
        )
        
        if hasattr(self.backbone_model, "patch_size"):
            model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(self.backbone_model.patch_size)(x[0]))
        
        model.init_weights()
        return model

