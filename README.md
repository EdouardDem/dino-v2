# DINOv2 Depth Estimation

## Docker commands

### Build image

Build the docker image:

```bash
docker build -t dino-v2-depth .
```

### Run server

Run the docker container:

```bash
docker run --rm \
    --gpus all \
    -p 8000:8000 \
    -e DEPTHER_BACKBONE_SIZE=small \
    -e DEPTHER_HEAD_TYPE=dpt \
    -e DEPTHER_HEAD_DATASET=nyu \
    --name depth-estimation \
    dino-v2-depth
```

#### Configuration Options

The following environment variables can be used to configure the depth estimation model:

- `DEPTHER_BACKBONE_SIZE`: Size of the backbone model (values `small`, `base`, `large` or `giant`, default: `small`)
- `DEPTHER_HEAD_TYPE`: Type of the depth estimation head (values `linear`, `linear4`, `dpt`, default: `dpt`)
- `DEPTHER_HEAD_DATASET`: Dataset used for the depth head (values `nyu`, `kitti`, default: `nyu`)

## Usage

### Usage with cURL

To generate a depth map from an image:

```bash
curl -X POST -F "file=@tests/dining-room.jpg" http://localhost:8000/depth/image --output tests/dining-room-depth-map.png
```

To generate a depth map video from a video file:

```bash
curl -X POST \
    -F "file=@tests/video.mp4" \
    -F "batch_size=4" \
    -F "scale_factor=1.0" \
    -F "fps=30" \
    http://localhost:8000/depth/video \
    --output tests/depth_video.mp4
```

Parameters for video processing:
- `batch_size`: Number of frames to process simultaneously (default: 4)
- `scale_factor`: Scale factor to apply to the video resolution (default: 1.0)
- `fps`: Output video frame rate (default: same as input video)

To check CUDA status:
```bash
curl http://localhost:8000/cuda-status
```

Example response for `/cuda-status`:
```json
{
    "cuda_available": true,
    "cuda_device_count": 1,
    "cuda_device_name": "NVIDIA GeForce RTX 3080"
}
```

The response indicates:
- `cuda_available`: whether CUDA is available on the system
- `cuda_device_count`: number of CUDA devices detected
- `cuda_device_name`: name of the first CUDA device (if available)

