import os
import glob
import scipy
from scipy.io import wavfile
import scipy.signal
import scipy.io
import numpy as np

import matplotlib.pyplot as plt
import librosa
import librosa.display
import noisereduce as nr
from datetime import datetime
from glob import glob

#Path to the Noise Files
path_noise = '.../Kettengeräusche'


#Read the noise files into an array audio_files
audio_files_noise = glob(path_noise + '/*.wav') #adds anthing with extension .wav
for i in range(len(audio_files_noise)):
    ynoise, srnoise = librosa.load(audio_files_noise[i], sr=44100)

#Define the time for name in the end
def extract_time_from_filename(filename):
    basename = os.path.basename(filename)
    datetime_str = os.path.splitext(basename)[0]  # remove extension
    datetime_obj = datetime.strptime(datetime_str, "%Y%m%d_%H%M%S")
    return datetime_obj.strftime("%Y%m%d_%H%M%S")


#Define the algorithm for loading the .wav files, splitting them if needed, remove the noise and saving them
def SpectroSplit(path_to_wav, path_to_save, splits=8, sr=44000, noise=0, mels=128, shift=0, cutoff=100):
    y, sr = librosa.load(path_to_wav, sr)
    b, a = scipy.signal.butter(3, [.02, .77], 'band')
    filteredBandPass = scipy.signal.lfilter(b, a, y) #Band-Pass filter
    #Attaching the file names to filenames
    time_info = extract_time_from_filename(path_to_wav)
    if noise.all() == 0:
        y_reduced = filteredBandPass
    else:
        y_reduced = nr.reduce_noise(y=filteredBandPass, sr=sr, y_noise=noise, stationary=False) #Non-stationary noise reduction
    y_red = nr.reduce_noise(y=y_reduced, sr=sr,
                                 freq_mask_smooth_hz=750, time_mask_smooth_ms=750) #Stationary Noise reduction
    if shift == 0:
        y_shift = y_red
    else:
        y_shift = librosa.effects.pitch_shift(y_red, sr, n_steps=shift, bins_per_octave=12)
    S = librosa.feature.melspectrogram(y=y_shift, sr=sr, n_mels=mels, fmin=10, fmax=sr/2)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB = S_dB[:,cutoff:(len(S_dB[0])-cutoff)]
    plt.close()
    for i in range(splits):
        S_dBs = S_dB[:,int((i/(splits+1))*len(S_dB[0])):int(((i+2)/(splits+1))*len(S_dB[0]))]
        librosa.display.specshow(S_dBs, sr=sr, fmin=10, fmax=sr/2)
        plt.savefig(f'{path_to_save}/{time_info}_Split{i}.png', pad_inches=0,
                bbox_inches='tight')
        plt.close()
    print(time_info + ' finished')

#Set path to .wav input files
path = ['J:/Rohdaten/2022-06-17/20220616*']

#Set savepath
savepath = 'E:/Split_3/sick'

#Load files
audio_files = []
for i in path:
    audio_files_i = glob(i + '.wav')
    audio_files.extend(audio_files_i)

#Check if files have been loaded
len(audio_files)


#Command to convert and split the files and save them
for i in range(len(audio_files)):
    SpectroSplit(audio_files[i], savepath, splits=3, sr=44000, noise=ynoise, mels=128, shift=0)

