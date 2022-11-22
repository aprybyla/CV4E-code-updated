# This code is to split my data into presence/absence classes for use in a binary classification CNN model

# PART 0: PACKAGES & PATHS 
# import packages 
import os
import numpy as np
import pandas as pd
import csv
import glob

from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram

# establish paths and veriables for data access
base_path = r'C:\Users\alixa\OneDrive\Desktop\Prototype_Data'
wav_files = os.listdir(os.path.join(base_path, r'.wav_files'))
flight_times = pd.read_csv(r'C:\Users\alixa\OneDrive\Desktop\Prototype_Data\flight_times.csv')
#wav_files_glob = glob.glob(r'C:\Users\alixa\OneDrive\Desktop\Prototype_Data\*.wav', recursive=True)
#print(wav_files_glob)
#print(wav_files)

# check to make sure your wav files are all there before you start
print(len(wav_files))

# recursively go through the wav files and 
audio_object = Audio.from_file(base_path, r'.wav_files')
print(f"How many samples does this audio object have? {len(audio_object.samples)}")
print(f"What is the sampling rate? {audio_object.sample_rate}")

audio_clips = []

clips, clip_df = audio_object.split(clip_duration=5,clip_overlap=0,final_clip=None)
print(f"duration of first clip: {clips[0].duration()}") 
clip_df.head(3)


#split and save
# for wav in wav_files: 
    #print(wav)
    # full_path = os.path.join(base_path, str(".wav_files/") + wav)
    # audio_segment = Audio.from_file(full_path,duration=0.5) 
    #audio_object.split(clip_duration=5,clip_overlap=0,final_clip=None)
    # audio_segments.append(audio_segment)
    #audio_segment.duration()

print(audio_clips)

print(audio_segment[1].duration())
print(audio_segment[2].duration())

#pandas lookup table (multifile from Carly, import csv and idex the labels)


# PART III: CREATING AND RANDOMIZING OUR CLASSES 
# define classes as bee presence/absence
#bee_presence_true=[]
#bee_presence_false=[]

# using permutation() method
#np.random.permutation(x)
#np.random.seed(0)
    #np.array([])
