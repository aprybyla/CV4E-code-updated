'''
    Loads audio files in (nested) folders and re-encodes them to the same sampling rate.
'''

import os
import glob
from tqdm import tqdm
from opensoundscape.audio import Audio

def standardize_sampling_rate(input_folder, sampling_rate=44100):
    '''TODO:
        1. Inputs:
            - path to files
            - target sampling rate
        2. Workflow:
            2.1 find all files
            2.2 for file in files:
            2.3   Audio.from_file(file, sampling_rate=...)
            2.4   Audio.save (?)
    '''
    # find all files
    all_files = glob.glob(os.path.join(input_folder, '**/*.wav'), recursive=True)

    for file_path in tqdm(all_files):
        audio_object = Audio.from_file(file_path, sampling_rate)
        assert audio_object.sample_rate == sampling_rate
        audio_object.save(file_path)

if __name__ == '__main__':
    #TODO
    standardize_sampling_rate(
        r'C:\Users\alixa\OneDrive\Desktop\Prototype_Data\Clipped_Audio',
        44100
    )