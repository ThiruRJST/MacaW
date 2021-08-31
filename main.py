import argparse
import librosa
import os
import h5py
from glob import glob
import numpy as np
from tqdm import tqdm
from spectro import wave_to_spec
import pandas as pd


def check_and_create(path):
    if os.path.isdir(path):
        print("Path already exists")
    else:
        os.makedirs(path)


def save_spec(path,offset=None,dur=None,save_format=".npy",save_path=None):
    check_and_create(save_path)
    for i in tqdm(range(len(path))):
        filename = path[i].split("/")[-1].split('.')[0]
        mels = wave_to_spec(path[i],offset=offset,duration=dur)
        if save_format == '.npy':
            np.save(os.path.join(save_path,filename+save_format),mels)
        elif save_format == '.h5':
            h5fil = h5py.File(os.path.join(save_path,filename+save_format),'w')
            h5fil.create_dataset('pixels',data=mels)
            h5fil.close()

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", help="The Path to the csv file.")
    parser.add_argument("--root_dir", help="The root dir of dataset.")
    parser.add_argument("--format", help="Data Format")
    parser.add_argument("--save_format",help="data format to save")
    parser.add_argument("--save_path",help="where to save")
    

    args = parser.parse_args()
    paths = args.csv_path
    save_as = args.save_format
    save_path = args.save_path
    root_dir = args.root_dir

    data = pd.read_csv(paths)
    #print(len(data))
    check_and_create(save_path)
    for n in tqdm(range(len(data))):
        rid = data.recording_id.iloc[n]
        p = os.path.join(root_dir,rid+args.format)
        mel = wave_to_spec(p,offset=data.t_min.iloc[n],duration=data.duration.iloc[n])
        if save_as == ".h5":
            h5fil = h5py.File(os.path.join(save_path,rid+save_as))
            h5fil.create_dataset("pixels",data=mel)
            h5fil.close()
        elif save_as == ".npy":
            np.save(os.path.join(root_dir,rid+save_as),arr=mel)