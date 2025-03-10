# From https://github.com/facebookresearch/dinov2/blob/main/notebooks/depth_estimation.ipynb

import math
import itertools
import urllib
import matplotlib
from functools import partial
from PIL import Image
import numpy as np
import cv2
from typing import Union, Tuple
import torch.utils.data
from pathlib import Path

import mmcv
from mmcv.runner import load_checkpoint

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from torchvision import transforms

from dinov2.eval.depth.models import build_depther

DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"

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


class Depther():

    cfg = None
    backbone_size = None
    head_type = None
    head_dataset = None
    backbone_model = None
    backbone_name = None
    model = None
    depth_transform = None

    def __init__(self, 
                 backbone_size = "small",  # in ("small", "base", "large" or "giant")
                 head_type = "dpt",  # in ("linear", "linear4", "dpt") 
                 head_dataset = "nyu"  # in ("nyu", "kitti")
                 ):
        self.backbone_size = backbone_size
        self.head_type = head_type
        self.head_dataset = head_dataset
        self.backbone_name = self._compute_backbone_name()

        self.cfg = self._get_head_config()
        self.backbone_model = self._load_backbone()
        self.model = self._create_depther()
        self._load_checkpoint()
        self.depth_transform = self._make_depth_transform()

    def get_depth_for_video(
        self, 
        video_path: Union[str, Path], 
        output_path: Union[str, Path],
        batch_size: int = 4,
        scale_factor: float = 1,
        fps: int = None
    ) -> None:
        """Process a video to generate depth estimation.
        
        Args:
            video_path: Path to the input video
            output_path: Path where to save the depth video
            batch_size: Number of frames to process simultaneously
            scale_factor: Scale factor to apply to the images
            fps: Frames per second for the output video. If None, uses the input video fps
        """
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
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path), 
            fourcc, 
            output_fps,
            (new_width, new_height),
            isColor=True
        )

        try:
            # Process video in batches
            current_batch = []
            
            for _ in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR frame to RGB and create PIL image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Resize if necessary
                if scale_factor != 1:
                    pil_image = pil_image.resize((new_width, new_height))
                
                # Add image to current batch
                current_batch.append(self.depth_transform(pil_image))
                
                # Process batch when full or at the end
                if len(current_batch) == batch_size or _ == total_frames - 1:
                    # Create batch tensor
                    batch_tensor = torch.stack(current_batch).cuda()
                    
                    # Process the batch
                    with torch.inference_mode():
                        results = self.model.whole_inference(batch_tensor, img_meta=None, rescale=True)
                    
                    # Process each result from the batch
                    for result in results:
                        depth_image = self._render_depth(result.cpu())
                        # Convert depth image to BGR for OpenCV
                        depth_frame = cv2.cvtColor(np.array(depth_image), cv2.COLOR_RGB2BGR)
                        out.write(depth_frame)
                    
                    # Reset batch
                    current_batch = []
        
        finally:
            # Release resources
            cap.release()
            out.release()

    def get_depth_for_image(self, image: Image.Image, scale_factor: float = 1) -> Image.Image:

        rescaled_image = image.resize((
            scale_factor * image.width, 
            scale_factor * image.height
        ))
        transformed_image = self.depth_transform(rescaled_image)
        batch = transformed_image.unsqueeze(0).cuda() # Make a batch of one image

        with torch.inference_mode():
            result = self.model.whole_inference(batch, img_meta=None, rescale=True)

        depth_image = self._render_depth(result.squeeze().cpu())
        
        return depth_image

    def _render_depth(self, values, colormap_name="magma_r") -> Image:
        min_value, max_value = values.min(), values.max()
        normalized_values = (values - min_value) / (max_value - min_value)

        colormap = matplotlib.colormaps[colormap_name]
        colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
        colors = colors[:, :, :3] # Discard alpha component
        return Image.fromarray(colors)

    def _make_depth_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.ToTensor(),
            lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
            transforms.Normalize(
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
            ),
        ])

    def _load_config_from_url(self, url: str) -> str:
        print(f"Loading config from {url}")
        with urllib.request.urlopen(url) as f:
            return f.read().decode()
    
    def _get_head_config(self):
        head_config_url = f"{DINOV2_BASE_URL}/{self.backbone_name}/{self.backbone_name}_{self.head_dataset}_{self.head_type}_config.py"

        cfg_str = self._load_config_from_url(head_config_url)
        cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

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


    def _create_depther(self):

        depther = build_depther(
            self.cfg.model,
            train_cfg=self.cfg.get("train_cfg"), 
            test_cfg=self.cfg.get("test_cfg")
        )

        depther.backbone.forward = partial(
            self.backbone_model.get_intermediate_layers,
            n=self.cfg.model.backbone.out_indices,
            reshape=True,
            return_class_token=self.cfg.model.backbone.output_cls_token,
            norm=self.cfg.model.backbone.final_norm,
        )

        if hasattr(self.backbone_model, "patch_size"):
            depther.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(self.backbone_model.patch_size)(x[0]))

        return depther
