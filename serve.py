from fastapi import FastAPI, UploadFile, File, Response
from PIL import Image
import io
from depther import Depther
import uvicorn
import torch
import os
from tempfile import NamedTemporaryFile
from fastapi.responses import FileResponse

app = FastAPI()

# Initialize Depther at startup
depther = Depther(
    backbone_size=os.getenv("DEPTHER_BACKBONE_SIZE", "small"),
    head_type=os.getenv("DEPTHER_HEAD_TYPE", "dpt"),
    head_dataset=os.getenv("DEPTHER_HEAD_DATASET", "nyu")
)

@app.get("/cuda-status")
async def get_cuda_status():
    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

@app.post("/depth/image")
async def process_image(file: UploadFile = File(...)):
    # Read image from uploaded file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Get depth map
    depth_image = depther.get_depth_for_image(image)
    
    # Convert depth image to bytes
    img_byte_arr = io.BytesIO()
    depth_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return Response(content=img_byte_arr, media_type="image/png")

@app.post("/depth/video")
async def process_video(
    file: UploadFile = File(...),
    batch_size: int = 4,
    scale_factor: float = 1.0,
    fps: int = None
):
    # Create temporary file for input video
    with NamedTemporaryFile(suffix=".mp4", delete=False) as input_video:
        # Write uploaded content to temporary file
        content = await file.read()
        input_video.write(content)
        input_video.flush()
        
        # Create temporary file for output
        with NamedTemporaryFile(suffix=".mp4", delete=False) as output_video:
            # Process the video
            depther.get_depth_for_video(
                video_path=input_video.name,
                output_path=output_video.name,
                batch_size=batch_size,
                scale_factor=scale_factor,
                fps=fps
            )
            
            # Delete input file
            os.unlink(input_video.name)
            
            # Return output file
            return FileResponse(
                output_video.name,
                media_type="video/mp4",
                filename="depth_" + file.filename,
                background=None  # To ensure file is deleted after sending
            )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 