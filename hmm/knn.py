from sklearn.neighbors import KNeighborsClassifier
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

n_sample = 20
data_len = [get_mfcc('data/len/len_{}.wav'.format(i)) for i in range(n_sample)]
data_xuong = [get_mfcc('data/xuong/xuong_{}.wav'.format(i)) for i in range(n_sample)]
data_trong = [get_mfcc('data/trong/trong_{}.wav'.format(i)) for i in range(n_sample)]
print(np.array(data_len[0]).shape)
X = data_len + data_xuong


# for x in X:
#     print(x)
y = []
for i in range(60):
    y.append(i//20)
print(y)

X = np.array(X)
print(X.shape)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

#with open("model/model_knn.pickle", "wb") as file: pkl.dump(model, file)