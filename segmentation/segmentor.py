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

from mmseg.apis import init_segmentor, inference_segmentor

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

    def get_depth_for_video(
        self, 
        video_path: Union[str, Path], 
        output_path: Union[str, Path],
        batch_size: int = 4,
        scale_factor: float = 1,
        fps: int = None,
        codec: str = 'mp4v',
        colormap_name: str = 'magma_r'
    ) -> None:
        """Process a video to generate depth estimation.
        
        Args:
            video_path: Path to the input video
            output_path: Path where to save the depth video
            batch_size: Number of frames to process simultaneously
            scale_factor: Scale factor to apply to the images
            fps: Frames per second for the output video. If None, uses the input video fps
            codec: Video codec to use ('avc1', 'h264' or 'mp4v', default: 'mp4v')
            colormap_name: Name of the matplotlib colormap to use (default: 'magma_r')
        """
        self._validate_colormap(colormap_name)

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
                        depth_image = self._render_segmentation(result.cpu(), colormap_name=colormap_name)
                        # Convert depth image to BGR for OpenCV
                        depth_frame = cv2.cvtColor(np.array(depth_image), cv2.COLOR_RGB2BGR)
                        out.write(depth_frame)
                    
                    # Reset batch
                    current_batch = []
        
        finally:
            # Release resources
            cap.release()
            out.release()

    def get_depth_for_image(
        self,
        image: Image.Image,
        colormap_name: str = 'magma_r'
    ) -> Image.Image:
        """Generate depth map for an image.
        
        Args:
            image: Input PIL image
            colormap_name: Name of the matplotlib colormap to use (default: 'magma_r')
            
        Returns:
            PIL Image containing the depth map
        """
        self._validate_colormap(colormap_name)
        result = inference_segmentor(self.model, image)[0]
        segmentation_image = self._render_segmentation(result.squeeze().cpu(), colormap_name=colormap_name)
        
        return segmentation_image

    def _render_segmentation(self, logits, colormap_name="magma_r") -> Image:
        colormap = matplotlib.colormaps[colormap_name]
        colormap_array = np.array(colormap, dtype=np.uint8)
        values = colormap_array[logits + 1]
        return Image.fromarray(values)

    def _validate_colormap(self, colormap_name: str) -> None:
        if colormap_name not in matplotlib.colormaps:
            raise ValueError(f"Invalid colormap name: {colormap_name}. Must be one of {', '.join(matplotlib.colormaps)}")


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

