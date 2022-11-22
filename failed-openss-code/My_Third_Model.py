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

test_df=pd.read_csv('/datadrive/' + "Clipped_Files_for_mc_Test.txt", sep=' ',
    header=None, names=["File_Name", "Presence_Absence"])
one_hot_labels_test, classes = categorical_to_one_hot(test_df[['Presence_Absence']].values)
test_df_labels = pd.DataFrame(index=test_df['File_Name'],data=one_hot_labels_test,columns=classes)
#test_df_labels.head()
#print(test_df.head())

#print("I'm printing the index", test_df.index)
#ipdb.set_trace()

train_df=pd.read_csv('/datadrive/' + "Clipped_Files_for_mc_Train.txt", sep=' ',
    header=None, names=["File_Name", "Presence_Absence"])
one_hot_labels_train, classes = categorical_to_one_hot(train_df[['Presence_Absence']].values)
train_df_labels = pd.DataFrame(index=train_df['File_Name'],data=one_hot_labels_train,columns=classes)
#print(train_df.head())

val_df=pd.read_csv('/datadrive/' + "Clipped_Files_for_mc_Val.txt", sep=' ',
    header=None, names=["File_Name", "Presence_Absence"])
one_hot_labels_val, classes = categorical_to_one_hot(val_df[['Presence_Absence']].values)
val_df_labels = pd.DataFrame(index=val_df['File_Name'],data=one_hot_labels_val,columns=classes)

model = CNN('resnet18',classes=classes,sample_duration=1.0,single_target=True)
print("model.single_target:", model.single_target)

#pick some random samples from the training set
sample_of_4 = test_df_labels.sample(n=4)

# #generate a dataset with the samples we wish to generate and the model's preprocessor
inspection_dataset = AudioFileDataset(sample_of_4, model.preprocessor)

#turn augmentation off for the dataset
# one of the defaults if you don't bypass augmentation is band preprocessing to 0-11.5kHz (currently running all preprocessors, see list)
inspection_dataset.bypass_augmentations = False

#generate the samples using the dataset
samples = [sample['X'] for sample in inspection_dataset]
labels = [sample['y'] for sample in inspection_dataset]

#display the samples
fig = show_tensor_grid(samples,4,labels=labels)
plt.savefig('/datadrive/' + "Practice_fig-mc.png")
plt.close()

model.verbose = 3 #how much content is printed
model.logging_level = 3 #request lots of logged content
model.log_file = './multilabel_train-15-08-2022/training_log-15-08-2022.txt' #specify a file to log output to
Path(model.log_file).parent.mkdir(parents=True,exist_ok=True) #make the folder

model.train(
    train_df=train_df_labels,
    validation_df=val_df_labels,
    save_path='./multilabel_train-15-08-2022/',
    epochs=50,
    batch_size=20,
    save_interval=50, #save model every 1 epoch
    num_workers=4
)

plt.scatter(model.loss_hist.keys(),model.loss_hist.values())
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('/datadrive/' + "Practice_model_loss_history-mc.png")