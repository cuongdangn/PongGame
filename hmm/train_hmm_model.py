import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import hmmlearn.hmm as hmm
from math import exp
import pickle as pkl

def get_mfcc(filename):
    data, fs = librosa.load(filename, sr=None)
    mfcc = librosa.feature.mfcc(data, sr=fs, n_fft=1024, hop_length=128)
    return mfcc.T

n_sample = 34
data_len = [get_mfcc('data/len1/len_{}.wav'.format(i)) for i in range(n_sample)]
data_xuong = [get_mfcc('data/xuong1/xuong_{}.wav'.format(i)) for i in range(n_sample)]
data_trong = [get_mfcc('data/dung/dung_{}.wav'.format(i)) for i in range(n_sample)]

model_len = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_len.fit(X=np.vstack(data_len), lengths=[x.shape[0] for x in data_len])

model_xuong = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_xuong.fit(X=np.vstack(data_xuong), lengths=[x.shape[0] for x in data_xuong])

model_trong = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_trong.fit(X=np.vstack(data_len), lengths=[x.shape[0] for x in data_trong])

with open("model/model_dung.pickle", "wb") as file: pkl.dump(model_trong, file)

with open("model/model_xuong1.pickle", "wb") as file: pkl.dump(model_xuong, file)

with open("model/model_len1.pickle", "wb") as file: pkl.dump(model_len, file)

