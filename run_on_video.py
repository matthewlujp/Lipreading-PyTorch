import os
import shutil
from argparse import ArgumentParser
import re
import toml
import imageio
from imageio.core.format import CannotReadFrameError
import cv2
import dlib
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
import torch.cuda as cuda
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

from models import LipRead
from data.preprocess import bbc


RIGHT_IDX = 3
LEFT_IDX = 13
rRIGHT_EYEBROW_IDX = 19
CHEEK_IDX = 9

FPS = 25
OUT_SIZE = 122


face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")


# with open("results/pretrained/jap2/options_used.toml", 'r') as f:
with open("results/pretrained/jap3/options_used.toml", 'r') as f:
    options = toml.loads(f.read())
model = LipRead(options)


def load_relevant_params(model, loaded_state_dict):
    # remove module
    state_dict = model.state_dict()
    for k, v in loaded_state_dict.items():
        res = re.match(r'(model).module.(.+)', k)
        if res:
            state_dict["{}.{}".format(res.group(1), res.group(2))] = v
        else:
            state_dict[k] = v
            
    model.load_state_dict(state_dict)


# state_dict = torch.load("results/pretrained/jap2/epoch29.pt", map_location=lambda storage, loc: storage)
state_dict = torch.load("results/pretrained/jap3/epoch29.pt", map_location=lambda storage, loc: storage)
load_relevant_params(model, state_dict)
model = model.eval()


words_list = sorted(os.listdir('train_data'))


def load_video(filename: str) -> list:
    """Load video from path and returns a list of frames as np.ndarrays.
    Return: [np.ndarray]
    """
    vid = imageio.get_reader(filename,  'ffmpeg')
    frames = []
    for i in range(0, 50):
        image = vid.get_data(i)
        image = np.array(image)
        frames.append(image)
    return frames



def get_facial_frames(frames: list) -> list:
    """Extract frames cropped around face area.
    Return: [np.ndarray]
    """
    ff = frames[0]
    ff_gray = cv2.cvtColor(ff, cv2.COLOR_RGB2GRAY)

    face_resion = face_detector(ff)
    if face_resion is None or len(face_resion) == 0:
        return None

    fpoints = shape_predictor(ff_gray, face_resion[0])
    # proc_frames = [crop_sq(f, fpoints, OUT_SIZE) for f in frames]
    proc_frames = frames  # if frames are already in 112x112 square
    return proc_frames


def crop_sq(image: np.ndarray, fpoints, out_length: int) -> np.ndarray:
    cx = int(np.mean([fpoints.part(RIGHT_IDX).x, fpoints.part(LEFT_IDX).x]))
    cy = int(np.mean([fpoints.part(RIGHT_EYEBROW_IDX).y, fpoints.part(CHEEK_IDX).y]))

    l = max(fpoints.part(LEFT_IDX).x - fpoints.part(RIGHT_IDX).x,
        fpoints.part(CHEEK_IDX).y - fpoints.part(RIGHT_EYEBROW_IDX).y)
    crop_l = int(l * 1.5)

    cropped = image[cy - crop_l//2:cy + (crop_l - crop_l//2), cx - crop_l//2:cx + (crop_l - crop_l//2)]
    resized = Image.fromarray(cropped).resize((out_length, out_length))
    return np.array(resized)


def get_frames(v_path: str) -> torch.FloatTensor:
    frames = load_video(v_path)  # [np.ndarray]
    face_frames = get_facial_frames([np.array(f) for f in frames])  # [np.ndarray]
    preproc_frames = bbc([torch.fromarray(f) for f in face_frames])  # torch.FloatTensor
    return preproc_frames
    

def visualize_confidence(model_output: torch.FloatTensor):
    # model_output: [frames, vocab]
    fig, ax = plt.subplots(figsize=(20, 5))
    heatmap = ax.pcolor(model_output.data.numpy(), cmap=plt.cm.Blues)

    ax.set_xticklabels(np.arange(0, model_output.shape[1], 10), fontsize=5)
    ax.set_xticks(np.arange(0, model_output.shape[1], 10))
    ax.set_yticks(np.arange(model_output.shape[0]), minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))

    plt.savefig('confidence.png')




if __name__ == '__main__':
    parser = ArgumentParser()    
    parser.add_argument("video_path")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("-v", "--visualize", action='store_true')
    args = parser.parse_args()

    frames = get_frames(args.video_path)
    if frames is None:
        raise Exception("failed to crop face")

    print("mean", torch.mean(frames), "std", torch.std(frames))

    # frames_preprocessed += - torch.mean(frames_preprocessed)
    # frames_preprocessed /= torch.std(frames_preprocessed)

    print("mean: {}, std: {}, frames length: {}".format(torch.mean(frames), torch.std(frames), frames.shape[1]))


    if args.save_dir is not None:
        if os.path.exists(args.save_dir):
            shutil.rmtree(args.save_dir)
        os.makedirs(args.save_dir) 
        for i, f in enumerate(frames.transpose(0, 1)):
            save_path = os.path.join(args.save_dir, "{}.png".format(i))
            F.to_pil_image(f, 'L').save(save_path)
    else:
        res = model(frames[None]).squeeze(0) # [frames, vocab]
        print("shape",res.shape)

        if args.visualize:
            visualize_confidence(res)

        # final prediction
        maxindices = torch.argmax(res)
        print(maxindices.data)
        if maxindices.data >= len(words_list):
            raise Exception("prediction out of word list")
        predicted = words_list[maxindices]
        print("final prediction => ", predicted)
