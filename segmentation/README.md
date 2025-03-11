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
    -F "colormap_name=magma_r" \
    http://localhost:8000/segmentation/image \
    --output tests/dining-room-segmentation-map.png
```

Parameters for image processing:
- `scale_factor`: Scale factor to apply to the image resolution (default: `1.0`)
- `colormap_name`: Matplotlib colormap to use for segmentation visualization (default: `magma_r`)

### Get segmentation map from video

To generate a segmentation map video from a video file:

```bash
curl -X POST \
    -F "file=@tests/video.mp4" \
    -F "scale_factor=1.0" \
    -F "fps=30" \
    -F "codec=mp4v" \
    -F "colormap_name=magma_r" \
    http://localhost:8000/segmentation/video \
    --output tests/segmentation_video.mp4
```

Parameters for video processing:
- `scale_factor`: Scale factor to apply to the video resolution (default: `1.0`)
- `fps`: Output video frame rate (default: same as input video)
- `codec`: Video codec to use (values `avc1`, `h264` or `mp4v`, default: `mp4v`)
- `colormap_name`: Matplotlib colormap to use for segmentation visualization (default: `magma_r`)

> [!NOTE]
> Only MP4 videos are supported. The H.264 codec (`avc1`) is recommended for better compatibility.

### Available Colormaps

The `colormap_name` parameter accepts any Matplotlib colormap name. Some popular options include:
- `magma_r` (default): Reversed magma colormap
- `viridis`: Green-blue colormap
- `plasma`: Plasma colormap
- `inferno`: Fire-like colormap
- `cividis`: Color-vision-deficiency-friendly colormap
- `binary`: Simple black and white colormap

For a complete list of available colormaps, visit the [Matplotlib colormap reference](https://matplotlib.org/stable/gallery/color/colormap_reference.html).

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

- `tests/segmentation_image_magma.png`: Segmentation map from the image with magma colormap
- `tests/segmentation_image_viridis.png`: Segmentation map from the image with viridis colormap
- `tests/segmentation_video.mp4`: Segmentation map from the video

> [!NOTE]
> This requires the Docker container to be running.

