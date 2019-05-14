import sounddevice as sd
import soundfile as sf
import numpy as np

def record_sound(filename, duration=1, fs=44100, play=False):
    print("start recored")
    while True:
        sd.play( np.sin( 2*np.pi*940*np.arange(fs)/fs )  , samplerate=fs, blocking=True)
        sd.play( np.zeros( int(fs*0.2) ), samplerate=fs, blocking=True)
        data = sd.rec(frames=duration*fs, samplerate=fs, channels=1, blocking=True)
        sd.play(data, samplerate=fs, blocking=True)
        print("It's ok ? (y/n):")
        a = input()
        if(a == 'y'):
            print("rc success")
            sf.write(filename, data=data, samplerate=fs)
            break
        else:
            print("rc again")


def record_data(prefix, n=20, duration=1):
    for i in range(n):
        print('{}_{}.wav'.format(prefix, i))
        record_sound('{}_{}.wav'.format(prefix, i), duration=duration)
        if i % 5 == 4:
            input("Press Enter to continue...")

record_data('data/trong/trong')