import os
import shutil
from argparse import ArgumentParser
import time
import re
import cv2
import toml
from PIL import Image
import dlib
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import imageio
from imageio.core.format import CannotReadFrameError

imageio.plugins.ffmpeg.download()

import torch
import torchvision.transforms.functional as F

from .data.preprocess import bbc, load_video
from .models import LipRead
# from data.preprocess import bbc, load_video
# from models import LipRead


face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(os.path.dirname(__file__) + "/shape_predictor_68_face_landmarks.dat")
# shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


RIGHT_IDX = 3
LEFT_IDX = 13
RIGHT_EYEBROW_IDX = 19
CHEEK_IDX = 9

FPS = 25
DEFAULT_DURATION = 1.6
OUT_SIZE = 256
ORG_WIDTH = 1280
ORG_HEIGHT = 720


with open(os.path.dirname(__file__) + "/results/pretrained/demo/options_used.toml", 'r') as f:
# with open("results/pretrained/demo/options_used.toml", 'r') as f:
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


# state_dict = torch.load(os.path.dirname(__file__) + "/results/pretrained/demo/epoch29.pt", map_location=lambda storage, loc: storage)
# state_dict = torch.load("results/pretrained/demo/epoch49.pt", map_location=lambda storage, loc: storage)
state_dict = torch.load(os.path.dirname(__file__) + "/results/pretrained/demo/epoch49.pt", map_location=lambda storage, loc: storage)
load_relevant_params(model, state_dict)
model = model.eval()


# words_list = sorted(os.listdir(os.path.dirname(__file__) + '/train_data'))
words_list = sorted(["anata", "ashita", "bhutan", "code", "demo", "doko", "dokusyo", "dou", "github", "hiromeru", "ichinichi", "itta", "kanazawa", "kanojo", "kenkyu", "konnichiwa",
    "kyo", "lab", "liptalk", "lunch", "meeting", "moshimoshi", "murakamiharuki", "nani", "none", "programming", "push", "saikin", "sakusei", "sekai", "shibuya", "shita", "sonoato",
    "syusyoku", "tabetai", "tanoshi", "totemo", "tsumaranai", "wo", "yojinbo", "yuhan", "yume"])

words_dict = {
    "anata": "あなた",
    "ashita": "あした",
    "bhutan": "ブータン",
    "code": "コード",
    "demo": "デモ",
    "doko": "どこ",
    "dokusyo": "どくしょ",
    "dou": "どう",
    "github": "ギットハブ",
    "hiromeru": "ひろめる",
    "ichinichi": "いちにち",
    "itta": "いった",
    "kanazawa": "かなざわ",
    "kanojo": "かのじょ",
    "kenkyu": "けんきゅう",
    "konnichiwa": "こんにちは",
    "kyo": "きょう",
    "lab": "ラボ",
    "liptalk": "リップトーク",
    "lunch": "ランチ",
    "meeting": "ミーティング",
    "moshimoshi": "もしもし",
    "murakamiharuki": "むらかみはるき",
    "nani": "なに",
    "none": "",
    "programming": "プログラミング",
    "push": "プッシュ",
    "saikin": "さいきん",
    "sakusei": "さくせい",
    "sekai": "せかい",
    "shibuya": "しぶや",
    "shita": "した",
    "sonoato": "そのあと",
    "syusyoku": "しゅうしょく",
    "tabetai": "たべたい",
    "tanoshi": "たのしい",
    "totemo": "とても",
    "tsumaranai": "つまらない",
    "wo": "を",
    "yojinbo": "ようじんぼう",
    "yuhan": "ゆうはん",
    "yume": "ゆめ"
}

def save_as_video(frames: list, save_path: str):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vwriter = cv2.VideoWriter(save_path, fourcc, FPS, (OUT_SIZE, OUT_SIZE))
    
    for frame in frames:
        vwriter.write(frame)
    
    vwriter.release()



def lipread(frames: list) -> str:
    """
    Args
    frames: [np.ndarray]

    Return: predicted word (str)
    """
    cropped_frames = process(frames)
    if cropped_frames is None:
        return "FACIAL DETECTION FAILED"

    t_cropped_frames = [F.to_tensor(f) for f in cropped_frames]
    preprocessed = bbc(t_cropped_frames, augmentation=False)

    res = model(preprocessed[None]).squeeze(0)

    maxindex = torch.argmax(res)
    return words_dict[words_list[maxindex]]

    
def process(frames: list) -> list:
    """
    Args
    frames: [np.ndarray]

    Return: cropped frames ([np.ndarray])
    """
    ff = frames[0]
    ff_gray = cv2.cvtColor(ff, cv2.COLOR_RGB2GRAY)

    face_resion = face_detector(ff_gray)
    if face_resion is None or len(face_resion) == 0:
        return None

    fpoints = shape_predictor(ff_gray, face_resion[0])
    proc_frames = [crop_sq(f, fpoints, OUT_SIZE) for f in frames]
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





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", dest="duration", type=float, default=DEFAULT_DURATION, help="video duration")
    args = parser.parse_args()

    print("------------------------------------------------------------------------------------------------------------")
    print("Press s in keyboard to start lipreading.\nRecording lasts for {} seconds.".format(args.duration))
    print("------------------------------------------------------------------------------------------------------------\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    cv2.namedWindow("Lipreading", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lipreading", 800, 450)

    max_frames = int(FPS * args.duration)
    record_frames = []
    recording_start_time = time.time()
    is_recording = False
    predicted_word = ""

    while cap.isOpened():
        ret, frame = cap.read() # 1280 x 720

        if is_recording:
            save_frame = np.copy(frame)
            record_frames.append(save_frame)

            elapsed_time = time.time() - recording_start_time
            cv2.putText(frame, "Lip reading   elapsed:{:.2}s".format(elapsed_time), (250, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 5)

            if len(record_frames) >= max_frames: # finish recording
                is_recording = False

                predicted_word = lipread(record_frames)
                print("predected word: {}".format(predicted_word))
                
        else:
            cv2.putText(frame, predicted_word, (300, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 5)

                
                
        cv2.imshow("Lipreading", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if not is_recording:
                record_frames = []
                is_recording = True
                recording_start_time = time.time()

            

        



    

    

    