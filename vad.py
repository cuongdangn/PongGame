'''
Requirements:
+ pyaudio - `pip install pyaudio`
+ py-webrtcvad - `pip install webrtcvad`
'''
import webrtcvad
import collections
import sys
import signal
import pyaudio

from array import array
from struct import pack
import wave
import time
import pickle as pkl
from math import exp
import librosa
import numpy as np

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION_MS = 30       # supports 10, 20 and 30 (ms)
PADDING_DURATION_MS = 1500   # 1 sec jugement
CHUNK_SIZE = (RATE * CHUNK_DURATION_MS // 1000)  # chunk to read
CHUNK_BYTES = CHUNK_SIZE * 2  # 16bit = 2 bytes, PCM
NUM_PADDING_CHUNKS = int(PADDING_DURATION_MS / CHUNK_DURATION_MS)
# NUM_WINDOW_CHUNKS = int(240 / CHUNK_DURATION_MS)
NUM_WINDOW_CHUNKS = int(400 / CHUNK_DURATION_MS)  # 400 ms/ 30ms  ge
NUM_WINDOW_CHUNKS_END = NUM_WINDOW_CHUNKS+20

START_OFFSET = int(NUM_WINDOW_CHUNKS * CHUNK_DURATION_MS * 0.5 * RATE)

vad = webrtcvad.Vad(1)

pa = pyaudio.PyAudio()
stream = pa.open(format=FORMAT,
                 channels=CHANNELS,
                 rate=RATE,
                 input=True,
                 start=False,
                 # input_device_index=2,
                 frames_per_buffer=CHUNK_SIZE)


got_a_sentence = False
leave = False
model_len = pkl.load(open('hmm/model/model_len1.pickle',"rb"))
model_xuong = pkl.load(open('hmm/model/model_xuong1.pickle',"rb"))
model_dung = pkl.load(open('hmm/model/model_dung.pickle',"rb"))
dic_index = {0:'l',1:'x', 2:'d'}

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.array(x-np.max(x))
    return e_x / e_x.sum(axis=0)

def handle_int(sig, chunk):
    global leave, got_a_sentence
    leave = True
    got_a_sentence = True


def record_to_file(path, data, sample_width):
    "Records from the microphone and outputs the resulting data to 'path'"
    # sample_width, data = record()
    data = pack('<' + ('h' * len(data)), *data)
    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 32767  # 16384
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)
    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r
def get_mfcc(filename):
    data, fs = librosa.load(filename, sr=None)
    mfcc = librosa.feature.mfcc(data, sr=fs, n_fft=1024, hop_length=128)
    return mfcc.T

def get_prob(log_x1, log_x2):
    if log_x1 < log_x2:
        exp_x1_x2 = exp(log_x1-log_x2)
        return exp_x1_x2 / (1+exp_x1_x2), 1 / (1+exp_x1_x2)
    else:
        p = get_prob(log_x2, log_x1)
        return p[1], p[0]

def process1():
    mfcc = get_mfcc("recording.wav")
    vote = [0,0,0]
    log_plen, log_pxuong, log_trong = model_len.score(mfcc), model_xuong.score(mfcc), model_dung.score(mfcc)
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
        return
    t = np.argmax(vote)
    
    f = open('command.txt','w')
    f.write(dic_index[t])
    f.close()

def process():
    mfcc = get_mfcc("recording.wav")
    log_plen, log_pxuong, log_dung = model_len.score(mfcc), model_xuong.score(mfcc), model_dung.score(mfcc)
    x = [log_plen, log_pxuong, log_dung]
    print(x)
    x = softmax(x)
    index = np.argmax(x)
    
    print(x[index], index, x)
    if(x[index]> 0.85):
        f = open('command.txt','w')
        f.write(dic_index[index])
        f.close()
    else:
        f = open('command.txt','w')
        f.write('d')
        f.close()
    

signal.signal(signal.SIGINT, handle_int)

data_len = 22
while not leave:
    ring_buffer = collections.deque(maxlen=NUM_PADDING_CHUNKS)
    triggered = False
    voiced_frames = []
    ring_buffer_flags = [0] * NUM_WINDOW_CHUNKS
    ring_buffer_index = 0

    ring_buffer_flags_end = [0] * NUM_WINDOW_CHUNKS_END
    ring_buffer_index_end = 0
    buffer_in = ''
    # WangS
    raw_data = array('h')
    index = 0
    start_point = 0
    StartTime = time.time()
    print("* recording: ")
    stream.start_stream()
    mystat = 0
    while not got_a_sentence and not leave:
        chunk = stream.read(CHUNK_SIZE)
        # add WangS
        raw_data.extend(array('h', chunk))
        index += CHUNK_SIZE
        TimeUse = time.time() - StartTime

        active = vad.is_speech(chunk, RATE)
        if active and mystat == 0:
            mystat = index-CHUNK_SIZE
        if not active:
            mystat = 0
        sys.stdout.write('*' if active else '_')
        ring_buffer_flags[ring_buffer_index] = 1 if active else 0
        ring_buffer_index += 1
        ring_buffer_index %= NUM_WINDOW_CHUNKS

        ring_buffer_flags_end[ring_buffer_index_end] = 1 if active else 0
        ring_buffer_index_end += 1
        ring_buffer_index_end %= NUM_WINDOW_CHUNKS_END

        # start point detection
        if not triggered:
            ring_buffer.append(chunk)
            num_voiced = sum(ring_buffer_flags)
            if num_voiced > 0.9 * NUM_WINDOW_CHUNKS:
                sys.stdout.write(' Open ')
                triggered = True
                #start_point = mystat - CHUNK_SIZE  # start point
                start_point = index - CHUNK_DURATION_MS*CHUNK_SIZE
                # voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
        # end point detection
        else:
            # voiced_frames.append(chunk)
            ring_buffer.append(chunk)
            num_unvoiced = NUM_WINDOW_CHUNKS_END - sum(ring_buffer_flags_end)
            if num_unvoiced > 0.8*NUM_WINDOW_CHUNKS_END:
                sys.stdout.write(' Close ')
                triggered = False
                raw_data.reverse()
                for index in range(start_point-1):
                    if len(raw_data) <= 0:
                        break
                    raw_data.pop()
                if len(raw_data) <= 0:
                        continue
                raw_data.reverse()
                raw_data = normalize(raw_data)
                record_to_file("recording.wav".format(data_len), raw_data, 2)
                data_len+=1
                ring_buffer.clear()
                index = 0
                start_point = 0
                raw_data = array('h')
                #process()
                ring_buffer = collections.deque(maxlen=NUM_PADDING_CHUNKS)
                triggered = False
                voiced_frames = []
                ring_buffer_flags = [0] * NUM_WINDOW_CHUNKS
                ring_buffer_index = 0

                ring_buffer_flags_end = [0] * NUM_WINDOW_CHUNKS_END
                ring_buffer_index_end = 0
                buffer_in = ''
                # WangS
                index = 0
                

        sys.stdout.flush()

    sys.stdout.write('\n')
    # data = b''.join(voiced_frames)

    stream.stop_stream()
    print("* done recording")
    got_a_sentence = False

    # write to file
    # raw_data.reverse()
    # for index in range(start_point):
    #     raw_data.pop()
    # raw_data.reverse()
    # raw_data = normalize(raw_data)
    # record_to_file("recording.wav", raw_data, 2)
    # leave = True

stream.close()
