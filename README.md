# DINOv2 Depth Estimation

## Usage

Build the docker image:

```bash
docker build -t dino-v2-depth .
```

Run the docker container:

```bash
docker run \
    --gpus all \
    -p 8000:8000 \
    dino-v2-depth
```

### Usage with cURL

```bash
curl -X POST -F "file=@tests/dining-room.jpg" http://localhost:8000/depth/image --output tests/dining-room-depth-map.png
```

