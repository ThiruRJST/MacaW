from torch.utils.data import Dataset,DataLoader
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import h5py
import librosa
import cv2
import torch
from tqdm import tqdm
import os
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

label_cols = ["species_"+str(m) for m in range(24)]

def h5read(path):
    hfile = h5py.File(path,'r')
    return np.array(hfile.get('pixels'))

def label_gen():
    label_dict = {}
    files = data.recording_id.unique().tolist()
    for f in files:
        labels = np.zeros(24)
        sets = group.get_group(f)
        tmp = sets.species_id.unique()
        for i in tmp:
            labels[i] = 1.
        
        label_dict[f] = labels
    return label_dict


class AudioData(Dataset):

    def __init__(self,records,targets,root_dir,transforms=None):
        
        self.root_dir = root_dir
        self.targets = targets
        self.records = records
        self.transforms = transforms
        
        #print(self.records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self,idx):
        
    
        img_arr = h5read(os.path.join(self.root_dir,self.records[idx]+'.h5'))
        label = self.targets[idx]

        assert img_arr.shape[2] == 3
        if self.transforms is not None:
            image = self.transforms(image=img_arr)['image']
        
        

        return image,torch.Tensor(label)


img = h5read("/home/lustbeast/AudioClass/Dataset/rfcx-species-audio-detection/h5/0a350d11c.h5")
print(img.shape)