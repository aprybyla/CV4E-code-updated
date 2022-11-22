#Python code to visualize/evaluate master list data

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\alixa\OneDrive\Desktop\Bee_Log_Master_Test.csv')

#filename = df['File Name'].to_numpy(dtype=str)
species = df['Species'].to_numpy(dtype=str)
caste = df['Caste'].to_numpy(dtype=str)

#assert len(filename) == len(species)
assert len(species) == len(caste)

counts = {}
filenames_done = []
for i in range(len(species)):
    id = species[i].strip() + '_' + caste[i].strip()
    if id not in counts:
        counts[id] = 0
    counts[id] += 1
print(counts)

# https://www.geeksforgeeks.org/python-remove-spaces-from-a-string/

import matplotlib.pyplot as plt

my_keys = []
my_values = []
for k in counts:
    my_keys.append(k)
    my_values.append(counts[k])

speciesnames = list(counts.keys())
numbernames = list(counts.values())
plt.bar(speciesnames,numbernames)
plt.xticks(rotation = 90)
plt.show()
