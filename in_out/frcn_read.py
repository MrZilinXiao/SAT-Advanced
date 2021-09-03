import numpy as np
from PIL import Image
import cv2
import torch
# disable cv2 multi-thread processing
cv2.setNumThreads(0)


def frcn_image_transform(image_path):
    img = Image.open(image_path)
    im = np.array(img).astype(np.float32)
    # handle a few corner cases
    if im.ndim == 2:  # gray => RGB
        im = np.tile(im[:, :, None], (1, 1, 3))
    if im.shape[2] > 3:  # RGBA => RGB
        im = im[:, :, :3]

    im = im[:, :, ::-1]  # RGB => BGR
    im -= np.array([102.9801, 115.9465, 122.7717])
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(800) / float(im_size_min)
    # Prevent the biggest axis from being more than max_size
    if np.round(im_scale * im_size_max) > 1333:
        im_scale = float(1333) / float(im_size_max)
    im = cv2.resize(
        im,
        None,
        None,
        fx=im_scale,
        fy=im_scale,
        interpolation=cv2.INTER_LINEAR
    )
    img = torch.from_numpy(im).permute(2, 0, 1)  # RGB
    return img, im_scale
