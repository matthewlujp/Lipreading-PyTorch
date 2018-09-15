import os
import shutil
from argparse import ArgumentParser
import re
from collections import OrderedDict
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
from data.preprocess import *


RIGHT_IDX = 3
LEFT_IDX = 13
RIGHT_EYEBROW_IDX = 19
CHEEK_IDX = 9

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")


with open("results/pretrained/jap4/options_used.toml", 'r') as f:
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


state_dict = torch.load("results/pretrained/jap4/epoch29.pt", map_location=lambda storage, loc: storage)
load_relevant_params(model, state_dict)
model = model.eval()


words_list = sorted(os.listdir('train_data'))



def preprocess(vpath) -> list:
    """
    Return: [torch.FloatTensor]
    """
    vframes = load_video(vpath)
    return bbc(vframes, augmentation=False)



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

    frames_preprocessed = preprocess(args.video_path)
    if frames_preprocessed is None:
        raise Exception("failed to crop face")

    
    # frames_preprocessed += - torch.mean(frames_preprocessed)
    # frames_preprocessed /= torch.std(frames_preprocessed)

    print("mean: {}, std: {}".format(torch.mean(frames_preprocessed), torch.std(frames_preprocessed)), frames_preprocessed.shape[1])


    if args.save_dir is not None:
        if os.path.exists(args.save_dir):
            shutil.rmtree(args.save_dir)
        os.makedirs(args.save_dir) 
        for i, f in enumerate(frames_preprocessed.transpose(0, 1)):
            save_path = os.path.join(args.save_dir, "{}.png".format(i))
            F.to_pil_image(f, 'L').save(save_path)
    else:
        res = model(frames_preprocessed[None]).squeeze(0) # [frames, vocab]
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
