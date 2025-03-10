from PIL import Image
import urllib.request

from depther import Depther


def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")


if __name__ == "__main__":
    image = load_image_from_url("https://dl.fbaipublicfiles.com/dinov2/images/example.jpg")
    model = Depther()
    depth_image = model.get_depth_for_image(image)
    depth_image.save("/app/output/depth_image.png")


