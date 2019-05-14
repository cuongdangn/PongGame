import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import hmmlearn.hmm as hmm
from math import exp
import pickle as pkl
from threading import Thread
model_len = pkl.load(open('hmm/model/model_len.pickle',"rb"))
model_xuong = pkl.load(open('hmm/model/model_xuong.pickle',"rb"))
model_trong = pkl.load(open('hmm/model/model_trong.pickle',"rb"))
dic_index = {0:'u',1:'d', 2:''}
def record_sound(filename, duration = 1, fs=44100, play=False):
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

while True:
    record_sound('test.wav')
    mfcc = get_mfcc('test.wav')
    vote = [0,0,0]
    log_plen, log_pxuong, log_trong = model_len.score(mfcc), model_xuong.score(mfcc), model_trong.score(mfcc)
    plen, pxuong = get_prob(log_plen, log_pxuong)
    if (plen > pxuong):
        vote[0] += 1
    else:
        vote[1] += 1
    plen, pxuong = get_prob(log_plen, log_trong)
    if (plen > pxuong):
        vote[0] += 1
    else:
        vote[2] += 1
    plen, pxuong = get_prob(log_trong, log_pxuong)
    if (plen > pxuong):
        vote[2] += 1
    else:
        vote[1] += 1
    if(vote[0] == vote[1] and vote[0] == 1):
        continue
    t = np.argmax(vote)
    
    f = open('command.txt','w')
    f.write(dic_index[t])
    f.close()