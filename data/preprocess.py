import os
import numpy as np
import cv2
import imageio

imageio.plugins.ffmpeg.download()

import torchvision.transforms.functional as functional
import torchvision.transforms as transforms
import torch
from .statefultransforms import StatefulRandomCrop, StatefulRandomHorizontalFlip


FRAMES_NUM = 40

def load_video(filename):
    """Loads the specified video using ffmpeg.

    Args:
        filename (str): The path to the file to load.
            Should be a format that ffmpeg can handle.

    Returns:
        List[FloatTensor]: the frames of the video as a list of 3D tensors
            (channels, width, height)"""

    vid = imageio.get_reader(filename,  'ffmpeg')
    frames = []
    for i in range(0, FRAMES_NUM):
        image = vid.get_data(i)
        image = functional.to_tensor(image)
        frames.append(image)
    return frames


def bbc(vidframes, augmentation=True):
    """Preprocesses the specified list of frames by center cropping.
    This will only work correctly on videos that are already centered on the
    mouth region, such as LRITW.

    Args:
        vidframes (List[FloatTensor]):  The frames of the video as a list of
            3D tensors (channels, width, height)

    Returns:
        FloatTensor: The video as a temporal volume, represented as a 5D tensor
            (batch, channel, time, width, height)"""

    temporalvolume = torch.FloatTensor(1,FRAMES_NUM,112,112)

    croptransform = transforms.CenterCrop((112, 112))

    if(augmentation):
        crop = StatefulRandomCrop((122, 122), (112, 112))
        flip = StatefulRandomHorizontalFlip(0.5)

        croptransform = transforms.Compose([
            crop,
            flip
        ])

    gray_frame = functional.to_tensor(functional.to_grayscale(functional.to_pil_image(vidframes[0])))
    mean = torch.mean(gray_frame)
    std = torch.std(gray_frame)

    for i in range(0, FRAMES_NUM):
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((122, 122)),
            croptransform,
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            # transforms.Normalize([0.4161,],[0.1688,]),
            transforms.Normalize([mean,],[std,]),
        ])(vidframes[i])

        temporalvolume[0][i] = result

    return temporalvolume
