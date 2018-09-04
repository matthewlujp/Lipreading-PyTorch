from torch.utils.data import Dataset
from .preprocess import *
import os

class LipreadingDataset(Dataset):
    """BBC Lip Reading dataset."""

    def build_video_file_list(self, dir: str, set: str):
        labels = sorted(os.listdir(dir))

        completeList = []

        for i, label in enumerate(labels):
            dirpath = dir + "/{}/{}".format(label, set)
            files = os.listdir(dirpath)
            for file in files:
                if file.endswith("mp4"):
                    filepath = dirpath + "/{}".format(file)
                    entry = (i, filepath)
                    completeList.append(entry)

        return labels, completeList

    def build_frames_file_list(self, dir: str, set: str):
        labels = sorted(os.listdir(dir))

        completeList = []

        for i, label in enumerate(labels):
            dirpath = dir + "/{}/{}".format(label, set)
            for v_name in os.listdir(dirpath):
                v_path = os.path.join(dirpath, v_name)
                entry = (i, v_path)
                completeList.append(entry)

        return labels, completeList

    def __init__(self, directory, set, augment=True, use_frames=False):
        if use_frames:
            self.label_list, self.file_list = self.build_frames_dir_list(directory, set)
        else:
            self.label_list, self.file_list = self.build_video_file_list(directory, set)
            
        self.augment = augment
        self.use_frames = use_frames

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """"load video into a tensor"""
        label, filename = self.file_list[idx]

        vidframes = load_frames(filename) if self.use_frames else load_video(filename)
            
        temporalvolume = bbc(vidframes, self.augment)

        sample = {'temporalvolume': temporalvolume, 'label': torch.LongTensor([label])}

        return sample
