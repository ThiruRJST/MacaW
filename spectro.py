import h5py
import librosa
import cv2
import numpy as np

class params:

    sampling_rate = 48000
    mel_bins = 128
    fmin = 20
    fmax = sampling_rate // 2

def mono_to_color(x,eps=1e-6):
   # X = np.stack([x,x,x],axis=-1)
    mean = x.mean()
    x = x-mean
    std = x.std()
    Xstd = X / std+eps
    _min,_max = Xstd.min(),Xstd.max()
    norm_max,norm_min = _max,_min
    if (_max - _min) > eps:
        V = Xstd
        V[V<norm_min] = norm_min
        V[V>norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(Xstd,dtype=np.uint8)
    
    return V

def wave_to_spec(path,offset=None,duration=None):
    
    audio,sr = librosa.load(path,sr=params.sampling_rate,offset=offset,duration=duration)
    melspec = librosa.feature.melspectrogram(y=audio,sr=sr,n_mels=params.mel_bins,fmin=params.fmin,fmax=params.fmax)
    melspec = librosa.power_to_db(melspec).astype(np.float32)
    melspec = mono_to_color(melspec)
    return melspec