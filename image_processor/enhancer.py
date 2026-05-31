# enhancer.py
import cv2
from realesrgan import RealESRGANer

upsampler = None

def enhance_image(img, scale=2):
    global upsampler
    if upsampler is None:
        upsampler = RealESRGANer(
            scale=scale,
            model_path='weights/RealESRGAN_x2plus.pth',
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True
        )

    output, _ = upsampler.enhance(img, outscale=scale)
    return output
