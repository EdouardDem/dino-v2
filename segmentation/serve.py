from fastapi import FastAPI, UploadFile, File, Response, Form
from PIL import Image
import io
from segmentor import Segmentor
import uvicorn
import torch
import os
from tempfile import NamedTemporaryFile
from fastapi.responses import FileResponse

app = FastAPI()

# Initialize Segmentor at startup
segmentor = Segmentor(
    backbone_size=os.getenv("SEGMENTOR_BACKBONE_SIZE", "small"),
    head_type=os.getenv("SEGMENTOR_HEAD_TYPE", "ms"),
    head_dataset=os.getenv("SEGMENTOR_HEAD_DATASET", "voc2012"),
    head_scale_count=int(os.getenv("SEGMENTOR_HEAD_SCALE_COUNT", "3"))
)

host = os.getenv("HOST", "0.0.0.0")
port = os.getenv("PORT", 8000)  

@app.get("/cuda-status")
async def get_cuda_status():
    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

@app.post("/segmentation/image")
async def process_image(
    file: UploadFile = File(...),
    scale_factor: float = Form(1.0),
    classes_only: str = Form(None),
):
    # Read image from uploaded file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Get segmentation map
    segmentation_image = segmentor.get_segmentation_for_image(
        image,
        scale_factor=scale_factor,
        classes_only=classes_only,
    )
    
    # Convert segmentation image to bytes
    img_byte_arr = io.BytesIO()
    segmentation_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return Response(content=img_byte_arr, media_type="image/png")

@app.post("/segmentation/video")
async def process_video(
    file: UploadFile = File(...),
    scale_factor: float = Form(1.0),
    fps: int = Form(None),
    codec: str = Form('mp4v'),
    classes_only: str = Form(None),
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
            segmentor.get_segmentation_for_video(
                video_path=input_video.name,
                output_path=output_video.name,
                scale_factor=scale_factor,
                fps=fps,
                codec=codec,
                classes_only=classes_only,
            )
            
            # Delete input file
            os.unlink(input_video.name)
            
            # Return output file
            return FileResponse(
                output_video.name,
                media_type="video/mp4",
                filename="segmentation_" + file.filename,
                background=None  # To ensure file is deleted after sending
            )

if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port) 