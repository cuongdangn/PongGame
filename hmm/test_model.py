import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import hmmlearn.hmm as hmm
import pickle as pkl
from math import exp

def record_sound(filename, duration=1, fs=44100, play=False):
    sd.play( np.sin( 2*np.pi*940*np.arange(fs)/fs )  , samplerate=fs, blocking=True)
    sd.play( np.zeros( int(fs*0.2) ), samplerate=fs, blocking=True)
    data = sd.rec(frames=duration*fs, samplerate=fs, channels=1, blocking=True)
    if play:
        sd.play(data, samplerate=fs, blocking=True)
    sf.write(filename, data=data, samplerate=fs)

def get_prob(log_x1, log_x2):
    if log_x1 < log_x2:
        exp_x1_x2 = exp(log_x1-log_x2)
        return exp_x1_x2 / (1+exp_x1_x2), 1 / (1+exp_x1_x2)
    else:
        p = get_prob(log_x2, log_x1)
        return p[1], p[0]

def get_mfcc(filename):
    data, fs = librosa.load(filename, sr=None)
    mfcc = librosa.feature.mfcc(data, sr=fs, n_fft=1024, hop_length=128)
    return mfcc.T

model_len = pkl.load(open('model/model_len.pickle',"rb"))
model_xuong = pkl.load(open('model/model_xuong.pickle',"rb"))
for i in range(15):
    record_sound('test.wav')
    mfcc = get_mfcc('test.wav')
    print(len(mfcc))
    log_plen, log_pxuong = model_len.score(mfcc), model_xuong.score(mfcc)
    plen, pxuong = get_prob(log_plen, log_pxuong)
    print(plen, pxuong, "len" if plen > pxuong else "xuong")