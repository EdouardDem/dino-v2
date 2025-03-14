# DINOv2 Segmentation

This is a simple FastAPI server that uses a pre-trained DINOv2 model to segment an image or video.

> [!IMPORTANT]
> This requires a GPU with CUDA support and the NVIDIA Container Toolkit.
> For more information, see the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Start server

### Run server using Docker

Run the docker container:

```bash
docker run --rm \
    --gpus all \
    -p 8000:8000 \
    -e SEGMENTOR_BACKBONE_SIZE=small \
    -e SEGMENTOR_HEAD_TYPE=ms \
    -e SEGMENTOR_HEAD_DATASET=voc2012 \
    -e SEGMENTOR_HEAD_SCALE_COUNT=3 \
    --name segmentation \
    edouarddem/dino-v2-segmentation
```

#### Configuration Options

The following environment variables can be used to configure the segmentation model:

- `SEGMENTOR_BACKBONE_SIZE`: Size of the backbone model (values `small`, `base`, `large` or `giant`, default: `small`)
- `SEGMENTOR_HEAD_TYPE`: Type of the segmentation head (values `ms`, `linear`, default: `ms`)
- `SEGMENTOR_HEAD_DATASET`: Dataset used for the segmentation head (values `ade20k`, `voc2012`, default: `voc2012`)
- `SEGMENTOR_HEAD_SCALE_COUNT`: Number of scales for multi-scale inference (values `1` to `5`, default: `3`)
- `HOST`: Host address to bind the server to (default: `0.0.0.0`)
- `PORT`: Port to bind the server to (default: `8000`)

### Run the server using docker-compose

```bash
docker-compose up -d
```

For more information, see the [docker-compose.yml](docker-compose.yml) file.

### Build image from source

Build the docker image:

```bash
docker build -t edouarddem/dino-v2-segmentation .
```

## Usage

### Get segmentation map from image

To generate a segmentation map from an image:

```bash
curl -X POST \
    -F "file=@tests/dining-room.jpg" \
    -F "scale_factor=1.0" \
    http://localhost:8000/segmentation/image \
    --output tests/dining-room-segmentation-map.png
```

Parameters for image processing:
- `scale_factor`: Scale factor to apply to the image resolution (default: `1.0`)

### Get segmentation map from video

To generate a segmentation map video from a video file:

```bash
curl -X POST \
    -F "file=@tests/video.mp4" \
    -F "scale_factor=1.0" \
    -F "fps=30" \
    -F "codec=mp4v" \
    http://localhost:8000/segmentation/video \
    --output tests/segmentation_video.mp4
```

Parameters for video processing:
- `scale_factor`: Scale factor to apply to the video resolution (default: `1.0`)
- `fps`: Output video frame rate (default: same as input video)
- `codec`: Video codec to use (values `avc1`, `h264` or `mp4v`, default: `mp4v`)

> [!NOTE]
> Only MP4 videos are supported. The H.264 codec (`avc1`) is recommended for better compatibility.

### Available Colormaps

Colormaps are predefined for the `ade20k` and `voc2012` datasets.
Each color is associated to a class label.

- `ade20k`: 150 classes
- `voc2012`: 21 classes

More information about the colormaps can be found in the [DINOv2 repository](https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/eval/segmentation/utils/colormaps.py).

## Check CUDA status

To get the GPU information:

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

## Client

We provide a simple client to get the segmentation map from an image or video.

The class `SegmentorClient` is defined in the [client.py](client.py) file.

You can test the client with the following command:

```bash
python client.py
```

This will create the following files:

- `tests/segmentation_image.png`: Segmentation map from the image
- `tests/segmentation_video.mp4`: Segmentation map from the video

> [!NOTE]
> This requires the Docker container to be running.

