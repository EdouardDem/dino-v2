import requests
from pathlib import Path
from typing import Optional, Union, Dict, Any
from PIL import Image
import io


class DeptherClient:
    """Client for interacting with the DINOv2 Depth Estimation server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client.
        
        Args:
            base_url: Base URL of the depth estimation server
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
    
    def get_depth_map(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> Union[Image.Image, None]:
        """Generate a depth map from an image.
        
        Args:
            image_path: Path to the input image
            output_path: Optional path to save the depth map. If not provided,
                        returns a PIL Image object
        
        Returns:
            PIL Image if output_path is None, else None
        """
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{self.base_url}/depth/image",
                files=files
            )
            response.raise_for_status()
            
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(response.content)
        else:
            return Image.open(io.BytesIO(response.content))
    
    def get_depth_video(
        self,
        video_path: Union[str, Path],
        output_path: Union[str, Path],
        batch_size: int = 4,
        scale_factor: float = 1.0,
        fps: Optional[int] = None,
        codec: str = 'mp4v'
    ) -> None:
        """Generate a depth map video from a video file.
        
        Args:
            video_path: Path to the input video
            output_path: Path to save the output video
            batch_size: Number of frames to process simultaneously
            scale_factor: Scale factor to apply to the video resolution
            fps: Output video frame rate (if None, uses input video fps)
            codec: Video codec to use (values `avc1`, `h264`, or `mp4v`, default: `mp4v`)
        """
        with open(video_path, 'rb') as f:
            files = {'file': f}
            data = {
                'batch_size': batch_size,
                'scale_factor': scale_factor
            }
            if fps is not None:
                data['fps'] = fps

            if codec is not None:
                data['codec'] = codec

            response = requests.post(
                f"{self.base_url}/depth/video",
                files=files,
                data=data
            )
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)


# Example usage
if __name__ == "__main__":
    client = DeptherClient("http://localhost:8000")
    
    # Check CUDA status
    cuda_status = client.get_cuda_status()
    print("CUDA Status:", cuda_status)
    
    # Process an image
    client.get_depth_map(
        "tests/image.jpg",
        "tests/depth_image.png"
    )
    
    # Process a video
    client.get_depth_video(
        "tests/video.mp4",
        "tests/depth_video.mp4",
        batch_size=4,
        scale_factor=1.0,
        fps=30,
        codec='mp4v'
    ) 
