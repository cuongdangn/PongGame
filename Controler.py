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
dic_index = {0:'l',1:'x', 2:'d'}
def record_sound(filename, duration = 1, fs=44100, play=False):
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

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def process1():
    record_sound('test.wav')
    mfcc = get_mfcc("test.wav")
    log_plen, log_pxuong, log_dung = model_len.score(mfcc), model_xuong.score(mfcc), model_trong.score(mfcc)
    x = [log_plen, log_pxuong, log_dung]
    x = softmax(x)
    index = np.argmax(x)
    if(x[index]> 0.85):
        f = open('command.txt','w')
        f.write(dic_index[index])
        f.close()
    else:
        f = open('command.txt','w')
        f.write('d')
        f.close()

while True:
    process1()
    # mfcc = get_mfcc('test.wav')
    # vote = [0,0,0]
    # log_plen, log_pxuong, log_trong = model_len.score(mfcc), model_xuong.score(mfcc), model_trong.score(mfcc)
    # plen, pxuong = get_prob(log_plen, log_pxuong)
    # if (plen > pxuong):
    #     vote[0] += 1
    # else:
    #     vote[1] += 1
    # plen, pxuong = get_prob(log_plen, log_trong)
    # if (plen > pxuong):
    #     vote[0] += 1
    # else:
    #     vote[2] += 1
    # plen, pxuong = get_prob(log_trong, log_pxuong)
    # if (plen > pxuong):
    #     vote[2] += 1
    # else:
    #     vote[1] += 1
    # if(vote[0] == vote[1] and vote[0] == 1):
    #     continue
    # t = np.argmax(vote)
    
    # f = open('command.txt','w')
    # f.write(dic_index[t])
    # f.close()