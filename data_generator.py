import os
import shutil
from argparse import ArgumentParser
import time
import re
import imageio
import cv2
from PIL import Image
import dlib
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from imageio.core.format import CannotReadFrameError

imageio.plugins.ffmpeg.download()


face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")


RIGHT_IDX = 3
LEFT_IDX = 13
RIGHT_EYEBROW_IDX = 19
CHEEK_IDX = 9

FPS = 25
DEFAULT_DURATION = 1.6
OUT_SIZE = 256
ORG_WIDTH = 1280
ORG_HEIGHT = 720


def process_and_save(frames: list, save_dir: str, word: str) -> str:
    save_path = os.path.join(save_dir, get_save_name(save_dir, word))
    proc_frames = process(frames)
    if proc_frames is None or len(proc_frames) == 0:
        print("failed to track face, abort")
        return None

    save_as_video(proc_frames, save_path)
    return save_path

    
def process(frames: list) -> list:
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


def save_as_video(frames: list, save_path: str):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vwriter = cv2.VideoWriter(save_path, fourcc, FPS, (OUT_SIZE, OUT_SIZE))
    
    for frame in frames:
        vwriter.write(frame)
    
    vwriter.release()


def get_save_name(save_dir, word) -> str:
    # decide video's serial number to fill worm-eaten state
    serial_numbers = []
    for filename in os.listdir(save_dir):
    
        res = re.match(word+r'_([0-9]{3}).mp4', filename)
        if res is None:
            continue
        serial_numbers.append(int(res.group(1)))

    for i, sn in enumerate(sorted(serial_numbers)):
        if i+1 != sn:
            video_name = "{}_{:03}.mp4".format(word, i+1)
            break
    else:
        video_name = "{}_{:03}.mp4".format(word, len(serial_numbers) + 1)

    return video_name
    
    
    


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("save_dir", help="directory to save data")
    parser.add_argument("word", help="videos are saved as word_001.mp4")
    parser.add_argument("-d", dest="duration", type=float, default=DEFAULT_DURATION, help="video duration")
    args = parser.parse_args()

    print("------------------------------------------------------------------------------------------------------------")
    print("Press s in keyboard to start recording.\nRecording lasts for {} seconds.".format(args.duration))
    print("An recorded video is automatically saved under {} in a format {}_001.mp4".format(args.save_dir, args.word))
    print("------------------------------------------------------------------------------------------------------------\n")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    cv2.namedWindow("Data generator: {}".format(args.word), cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Data generator: {}".format(args.word), 800, 450)

    max_frames = int(FPS * args.duration)
    record_frames = []
    recording_start_time = time.time()
    is_recording = False

    while cap.isOpened():
        ret, frame = cap.read() # 1280 x 720

        if is_recording:
            save_frame = np.copy(frame)
            record_frames.append(save_frame)

            elapsed_time = time.time() - recording_start_time
            cv2.putText(frame, "RECORDING   elapsed:{:.2}s".format(elapsed_time), (250, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 5)

            if len(record_frames) >= max_frames: # finish recording
                is_recording = False
                save_path = process_and_save(record_frames, args.save_dir, args.word)
                print("a sample for {} is saved in {}".format(args.word, save_path))
                
                
        cv2.imshow("Data generator: {}".format(args.word), frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if not is_recording:
                record_frames = []
                is_recording = True
                recording_start_time = time.time()

            

        



    

    

    