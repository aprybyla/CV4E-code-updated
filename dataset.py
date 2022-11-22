import os
import json
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import numpy as np
import torchaudio
import torch
import torchvision

# Here, we define the functionget_split_index that dictates how the splits are going to happen later in the code (does not happen right away)
# We will call this function that we have defined later on, and pass that function our data, and it will run this chunk on our data. This just defines
# what will happen when we call the function. 

# num_items is the number of items in the split. This includes the whole dataset, so what will eventually become the train, test, and val sets. 
# frac is the fraction that we will set for each setting (so 70% for train, 15% for test and val). This would be Val_frac = 15, &c. 
# seed = 1234, this allows us to keep the way the data is split the same every single time. So the same files wind up in the same sets every time. 
def get_split_idx(num_items, frac, seed=1234):
    '''
        Generates indices for splitting a dataset. 
    '''
    # compute size of each split:
    num_split_1 = int(np.round((1.0 - frac) * num_items)) # number of items in first split
    num_split_2 = num_items - num_split_1 # number of items in second split
    # get indices for each split:
    rng = np.random.default_rng(seed) # Want the split to be the same every time.
    idx_rand = rng.permutation(num_items) # Permutation of 0, 1, ..., num_items-1
    idx_split_1 = idx_rand[:num_split_1]
    idx_split_2 = idx_rand[-num_split_2:]
    assert len(idx_split_1) + len(idx_split_2) == num_items # check that the split sizes add up
    return idx_split_1, idx_split_2

# min_max_normalizes normalizes the spectrograms. It regularizes our spectrogram inputs so that they are between 0 and 1 (a known scale)
def min_max_normalize(x):
    x = x - torch.min(x)
    x = x / torch.max(x)
    return x

# here we are defining the class AudioDataset. This is the dataloader. This is how we load the audio in and make it spectrograms. It creates the input to the model
# Between the data and the model, this is the bridge. 
class AudioDataset(Dataset):

    # this is saying, for the train data, do these trainsforms (which will come later)
    # We write the definitions of stuff first, and then we call them later. This is the "preprocessor pipeline"
    def __init__(self, cfg, split='train'):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = cfg['data_root']
        self.split = split
        
        #CLASS ORGANIZATION FOR SPECIES-CASTE LABELS 
        self.class_mapping = {}
        self.class_mapping['Blank'] = 0
        self.class_mapping['Anthophora-plumipes-nan'] = 1
        self.class_mapping['Bombus-terrestris-queen'] = 2
        self.class_mapping['Bombus-terrestris-worker'] = 3
        self.class_mapping['Bombus-pratorum-queen'] = 4
        self.class_mapping['Bombus-pratorum-worker'] = 5
        self.class_mapping['Bombus-hypnorum-queen'] = 6
        self.class_mapping['Bombus-hypnorum-worker'] = 7
        self.class_mapping['Bombus-pascuorum-queen'] = 8

        #CLASS ORGANIZATION FOR CASTE LABELS 
        # self.class_mapping = {}
        # self.class_mapping['Blank'] = 0
        # self.class_mapping['Anthophora-plumipes-nan'] = 3
        # self.class_mapping['Bombus-terrestris-queen'] = 1
        # self.class_mapping['Bombus-terrestris-worker'] = 2
        # self.class_mapping['Bombus-pratorum-queen'] = 1
        # self.class_mapping['Bombus-pratorum-worker'] = 2
        # self.class_mapping['Bombus-hypnorum-queen'] = 1
        # self.class_mapping['Bombus-hypnorum-worker'] = 2
        # self.class_mapping['Bombus-pascuorum-queen'] = 1

        # CLASS ORGANIZATION FOR SPECIES-ONLY LABELS 
        # self.class_mapping = {}
        # self.class_mapping['Blank'] = 0
        # self.class_mapping['Anthophora-plumipes-nan'] = 1
        # self.class_mapping['Bombus-terrestris-queen'] = 2
        # self.class_mapping['Bombus-terrestris-worker'] = 2
        # self.class_mapping['Bombus-pratorum-queen'] = 3
        # self.class_mapping['Bombus-pratorum-worker'] = 3
        # self.class_mapping['Bombus-hypnorum-queen'] = 4
        # self.class_mapping['Bombus-hypnorum-worker'] = 4
        # self.class_mapping['Bombus-pascuorum-queen'] = 5

        # define transforms (ways to preprocess):
        # Note - this changes the sampling rate, not the Nyquist frequency. So 16,000 would be 8,000 in principal 
        # This is something I can per-project define and mess around with 
        resamp = torchaudio.transforms.Resample(
            orig_freq=44100,
            new_freq=16000
            )
        # This is window size and stuff for spectrograms 
        # Default window shape parameter, using Torch.audio thing. We can mess with the window shape functionality by 
        # making it window shape = Hann and window shape = Hamming, &c. 
        to_spec = torchaudio.transforms.MelSpectrogram(
            n_fft=512,
            hop_length=128 #pretty default 
            )
        time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=60, # mask up to 60 consecutive time windows
        )
        freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=8, # mask up to 8 consecutive frequency bins
        )
        #if we are working on the train dataset, we are going to do all of the transforms, because we want to do the augmentation on the training and not the validation set. 
        # So the time and frequency mask is listed for the train, but not the else which would be the validation 
        if split == 'train':
            self.transform = Compose([                              # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
                resamp,                                             # resample to 16 kHz
                to_spec,                                            # convert to a spectrogram
                torchaudio.transforms.AmplitudeToDB(),
                torchvision.transforms.Lambda(min_max_normalize),   # normalize so min is 0 and max is 1
                time_mask,                                          # AUGMENTATION randomly mask out a chunk of time
                freq_mask,                                          # AUGMENTATION randomly mask out a chunk of frequencies
                Resize(cfg['image_size']),
            ])
        else:
            self.transform = Compose([                              # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
                resamp,                                             # resample to 16 kHz
                to_spec,                                            # convert to a spectrogram
                torchaudio.transforms.AmplitudeToDB(),
                torchvision.transforms.Lambda(min_max_normalize),   # normalize so min is 0 and max is 1
                Resize(cfg['image_size']),
            ])


################# NOW WE LOAD OUR ANNOTATIONS THAT MAP ONTO THE AUDIO ########################################################

        if split == 'train':
            annoPath = os.path.join(self.data_root, 'Clipped_Files_for_mc_Train.txt')
        elif split == 'val': 
            annoPath = os.path.join(self.data_root, 'Clipped_Files_for_mc_Val.txt')
        elif split == 'test': 
            annoPath = os.path.join(self.data_root, 'Clipped_Files_for_mc_Test.txt')

        # load annotation file
        # annoPath = os.path.join(
        #     self.data_root,
        #     'audio_ml.csv' # This is something I would change to whatever my directory is 
        # )
        with open(annoPath, 'r') as f: 
            csv_lines = f.readlines()
        #csv_lines = csv_lines[1:] # get rid of the header row
        csv_lines = [l.rstrip() for l in csv_lines] # delete newline character (\n) from the end of each line
        
        # get the filenames and labels for the non-test data we're allowed to use for model development:
        # Here is where we efine the columns and the classes
        dev_filenames = []
        dev_labels = []
        for l in csv_lines:
            # split out the fields in the current row of the csv:
            asset_id, label = l.split(' ') # COLUMBS FROM EXAMPLE DATA 
            # Note: From the dataset documentation, we know that all of the audio files have 1 channel, are 10s long, and have a sample rate of 22.05 kHz. But let's check those assumptions.
            # assert int(channels) == 1
            # assert float(duration_seconds) == 1.0 #we have set this to one second because thats how long my clips are
            # assert float(samplerate) == 22.05 * 1000
            # if split_assignment == 'train':
            dev_filenames.append(asset_id) #what is written here as asset.id is what we have as "File_Name", a list of file names. 
            dev_labels.append(self.class_mapping[label]) 
           
        # Here, we assign our labels (also known as classes)
        dev_filenames = np.array(dev_filenames)
        dev_labels = np.array(dev_labels)
        
        # SSW60 does not have an official val set, so we create one by taking some of the training data:
        # We are now using the get split idx function above to separate our classes
        # We are using to fractionate/define how we want to split our percentage 
        # val_frac = 0.15
        # print('Creating {:.0f}/{:.0f} split (train/val). Choosing {}.'.format(100*(1-val_frac), 100*val_frac, split))
        # idx_train, idx_val = get_split_idx(len(dev_filenames), val_frac)
        
        # pick filenames and labels based on the split we're currently working with:
        # FOR THE PURPOSES OF THIS DATASET, THE DATA HAS ALREADY BEEN SPLIT AND LIVES SOMEWHERE ELSE 
    
        filenames = dev_filenames
        labels = dev_labels
        
        
        # index data into list (IF YOU HAVE ALREADY MANYALLY DONE THIS THEN YOU DONT HAVE TO USE THIS PART OF THE CODE)
        self.data = []
        for filename, label in zip(filenames, labels):
            self.data.append([filename, label])
    

    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)

    
    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx.
            Here's where we actually load the image.
        '''
        image_name, label = self.data[idx]

        # load image
        audio_path = os.path.join(self.data_root, image_name) # we are constructing a path from what is written here as audio_ml
        waveform, sample_rate = torchaudio.load(audio_path)
        image = self.transform(waveform)
        image = image.expand(3, -1, -1) # replicate to 3 channels (CNNs expect 3 channels fo input, like RGB. But we only have one channel.)

        return image, label