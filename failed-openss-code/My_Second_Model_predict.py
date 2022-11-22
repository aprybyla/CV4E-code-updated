# the cnn module provides classes for training/predicting with various types of CNNs
from opensoundscape.torch.models.cnn import CNN

#other utilities and packages
import os 
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
import random
import subprocess
import ipdb

#set up plotting
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize']=[15,5] #for large visuals

#helper functions to visualize processed samples
from opensoundscape.preprocess.utils import show_tensor_grid, show_tensor
from opensoundscape.torch.datasets import AudioFileDataset
from opensoundscape.torch.models.cnn import load_model
from opensoundscape.annotations import categorical_to_one_hot

base_path = './'
class_label = "class_label"

test_df=pd.read_csv('./' + "Clipped_Files_for_mc_Test.txt", sep=' ',
    header=None, names=["File_Name", class_label])
one_hot_labels_test, classes = categorical_to_one_hot(test_df[[class_label]].values)

#model = CNN('resnet18',classes=classes,sample_duration=1.0,single_target=True)
model = load_model('./multilabel_train/best.model')

import glob
files = glob.glob('./Clipped_Audio_for_Binary_Classification_TEST/*')
labels = np.argmax(one_hot_labels_test, axis=1)
#print (files[0])
for i,test_file in enumerate(files):
    print (test_file)
    print ('\n')
    #test_file = ['./Clipped_Audio_for_Binary_Classification_TEST/29-03-2022_RBGE-013_B-pratorum_queen_BEE-PRESENT_Audacity.wav.wav_0.8987074829931974_1.8987074829931974.wav']
    scores,_,_ = model.predict([test_file], activation_layer='softmax')
    ground_truth_class = test_df.loc[test_df['File_Name'] == test_file]['class_label'].values[0]
    print ('ground truth class is ' + ground_truth_class + ' ')
    print ('prediction is ')
    max_score = 0.0
    max_class = ''
    if scores.empty:
        print ('!!!!!!!!!!!!!!!!!!!!! SKKKIIPPPP !!!!!' + test_file)
        continue
             
    for cl in classes:
        score = scores[cl].values[0]
        if score > max_score:
            max_score = score
            max_class = cl
    print (max_class, max_score)
    #ipdb.set_trace()
    #break

#print (scores)
#import ipdb; ipdb.set_trace()
#print (scores)
