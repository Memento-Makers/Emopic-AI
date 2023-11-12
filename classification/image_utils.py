import requests
import torchvision
from PIL import Image as PIL_Image


def preprocess_image(image_path, img_size):
    pil_image = PIL_Image.open(requests.get(image_path, stream=True).raw)
    pil_image = pil_image.resize((img_size,img_size))
    return pil_image