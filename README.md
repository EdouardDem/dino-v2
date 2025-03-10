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
    --name depth-estimation \
    dino-v2-depth
```

## Usage

### Usage with cURL

To generate a depth map from an image:
```bash
curl -X POST -F "file=@tests/dining-room.jpg" http://localhost:8000/depth/image --output tests/dining-room-depth-map.png
```

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

