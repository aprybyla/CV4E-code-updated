######################################################################################
# This code is to help Alix count individuals in her dataset and group them by caste #
# Code was written by Eli Cole at CalTech and Alixandra Prybyla at Uni of Edinburgh  #
######################################################################################

import pandas as pd

df = pd.read_csv(r'C:\Users\alixa\OneDrive\Desktop\Prototype Data\flight_times.csv')
#df = pd.read_csv(r'C:\Users\alixa\OneDrive\Desktop\Bee_Log_Master_Test.csv')

filename = df['File Name'].to_numpy(dtype=str)
species = df['Species'].to_numpy(dtype=str)
caste = df['Caste'].to_numpy(dtype=str)

assert len(filename) == len(species)
assert len(species) == len(caste)

counts = {}
filenames_done = []
for i in range(len(filename)):
    if filename[i] in filenames_done:
        continue
    id = species[i] + '_' + caste[i]
    if id not in counts:
        counts[id] = 0
    counts[id] += 1
    filenames_done.append(filename[i])
print(counts)

