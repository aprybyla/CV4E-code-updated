# the cnn module provides classes for training/predicting with various types of CNNs
from opensoundscape.torch.models.cnn import CNN

#other utilities and packages
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
import random
import subprocess

#set up plotting
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize']=[15,5] #for large visuals

#helper functions to visualize processed samples
from opensoundscape.preprocess.utils import show_tensor_grid, show_tensor
from opensoundscape.torch.datasets import AudioFileDataset

from opensoundscape.annotations import categorical_to_one_hot

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

test_df=pd.read_csv('/mnt/c/Users/alixa/OneDrive/Desktop/Prototype_Data/' + "Clipped_Files_for_bc_Test.txt", sep=' ',
    header=None, names=["File_Name", "Presence_Absence"])
one_hot_labels_test, classes = categorical_to_one_hot(test_df[['Presence_Absence']].values)
test_df_labels = pd.DataFrame(index=test_df['File_Name'],data=one_hot_labels_test,columns=classes)
#test_df_labels.head()
#print(test_df.head())

#print("I'm printing the index", test_df.index)
#ipdb.set_trace()

train_df=pd.read_csv('/mnt/c/Users/alixa/OneDrive/Desktop/Prototype_Data/' + "Clipped_Files_for_bc_Train.txt", sep=' ',
    header=None, names=["File_Name", "Presence_Absence"])
one_hot_labels_train, classes = categorical_to_one_hot(train_df[['Presence_Absence']].values)
train_df_labels = pd.DataFrame(index=train_df['File_Name'],data=one_hot_labels_train,columns=classes)
#print(train_df.head())

val_df=pd.read_csv('/mnt/c/Users/alixa/OneDrive/Desktop/Prototype_Data/' + "Clipped_Files_for_bc_Val.txt", sep=' ',
    header=None, names=["File_Name", "Presence_Absence"])
one_hot_labels_val, classes = categorical_to_one_hot(val_df[['Presence_Absence']].values)
val_df_labels = pd.DataFrame(index=val_df['File_Name'],data=one_hot_labels_val,columns=classes)
#print(val_df.head())

model = CNN('resnet18',classes=classes,sample_duration=1.0,single_target=True)

#pick some random samples from the training set
sample_of_4 = test_df_labels.sample(n=4)

#generate a dataset with the samples we wish to generate and the model's preprocessor
inspection_dataset = AudioFileDataset(sample_of_4, model.preprocessor)

#turn augmentation off for the dataset - time frequency masking
inspection_dataset.bypass_augmentations = False

#generate the samples using the dataset
samples = [sample['X'] for sample in inspection_dataset]
labels = [sample['y'] for sample in inspection_dataset]

#display the samples
fig = show_tensor_grid(samples,4,labels=labels)
plt.savefig('/mnt/c/Users/alixa/OneDrive/Desktop/Prototype_Data/' + "Practice_fig.png")
plt.close()

model.train(
    train_df=train_df_labels,
    validation_df=val_df_labels,
    save_path='./binary_train/', #where to save the trained model
    epochs=100,
    batch_size=64,
    save_interval=5, #save model every 5 epochs (the best model is always saved in addition)
    num_workers=4, #specify 4 if you have 4 CPU processes, eg; 0 means only the root process
)

plt.scatter(model.loss_hist.keys(),model.loss_hist.values())
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('/mnt/c/Users/alixa/OneDrive/Desktop/Prototype_Data/' + "Practice_model_loss_history.png")