import requests
from pathlib import Path
from typing import Optional, Union, Dict, Any
from PIL import Image
import io


class SegmentorClient:
    """Client for interacting with the DINOv2 Segmentation server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client.
        
        Args:
            base_url: Base URL of the segmentation server
        """
        self.base_url = base_url.rstrip('/')
        
    def get_cuda_status(self) -> Dict[str, Any]:
        """Get the CUDA status from the server.
        
        Returns:
            Dict containing CUDA information:
            - cuda_available: whether CUDA is available
            - cuda_device_count: number of CUDA devices
            - cuda_device_name: name of the first CUDA device
        """
        response = requests.get(f"{self.base_url}/cuda-status")
        response.raise_for_status()
        return response.json()
    
    def get_segmentation_map(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        scale_factor: float = 1.0,
        colormap_name: str = 'magma_r'
    ) -> Union[Image.Image, None]:
        """Generate a segmentation map from an image.
        
        Args:
            image_path: Path to the input image
            output_path: Optional path to save the segmentation map. If not provided,
                        returns a PIL Image object
            scale_factor: Scale factor to apply to the image resolution
            colormap_name: Name of the matplotlib colormap to use (default: 'magma_r')
        
        Returns:
            PIL Image if output_path is None, else None
        """
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'scale_factor': scale_factor,
                'colormap_name': colormap_name
            }
            response = requests.post(
                f"{self.base_url}/segmentation/image",
                files=files,
                data=data
            )
            response.raise_for_status()
            
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(response.content)
        else:
            return Image.open(io.BytesIO(response.content))
    
    def get_segmentation_video(
        self,
        video_path: Union[str, Path],
        output_path: Union[str, Path],
        scale_factor: float = 1.0,
        fps: Optional[int] = None,
        codec: str = 'mp4v',
        colormap_name: str = 'magma_r'
    ) -> None:
        """Generate a segmentation map video from a video file.
        
        Args:
            video_path: Path to the input video
            output_path: Path to save the output video
            scale_factor: Scale factor to apply to the video resolution
            fps: Output video frame rate (if None, uses input video fps)
            codec: Video codec to use ('avc1' for H.264 or 'mp4v', default: 'mp4v')
            colormap_name: Name of the matplotlib colormap to use (default: 'magma_r')
        """
        with open(video_path, 'rb') as f:
            files = {'file': f}
            data = {
                'scale_factor': scale_factor,
                'codec': codec,
                'colormap_name': colormap_name
            }
            if fps is not None:
                data['fps'] = fps
                
            response = requests.post(
                f"{self.base_url}/segmentation/video",
                files=files,
                data=data
            )
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)


# Example usage
if __name__ == "__main__":
    client = SegmentorClient("http://localhost:8000")
    folder = Path(__file__).parent.parent / "tests"
    
    # Check CUDA status
    cuda_status = client.get_cuda_status()
    print("CUDA Status:", cuda_status)
    
    # Process an image with different colormaps
    client.get_segmentation_map(
        folder / "image.jpg",
        folder / "segmentation_image_magma.png",
        colormap_name='magma_r'
    )
    
    client.get_segmentation_map(
        folder / "image.jpg",
        folder / "segmentation_image_viridis.png",
        colormap_name='viridis'
    )
    
    # Process a video
    client.get_segmentation_video(
        folder / "video.mp4",
        folder / "segmentation_video.mp4",
        scale_factor=1.0,
        fps=30,
        codec='mp4v',
        colormap_name='binary'
    ) 
