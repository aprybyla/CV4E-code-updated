#This code is to split my data by name only. 

import math
import os
import pandas as pd
import torch 
import math

base_path = r'C:\Users\alixa\OneDrive\Desktop\Prototype_Data'

wav_files = os.listdir(os.path.join(base_path, r'.wav_files'))

# print(wav_files)
anthophora_plumipes_nan_class=[]
bombus_terrestris_queen_class=[]
bombus_terrestris_worker_class=[]
bombus_pratorum_queen_class=[]
bombus_pratorum_worker_class=[]
bombus_hypnorum_queen_class=[]
bombus_hypnorum_worker_class=[]
bombus_pascuorum_queen_class=[]
for i in wav_files:
    g=i.split('_')
    pt1=g[2]
    pt2=g[3]
    # if pt1=='B-pascuorum':
    #     print(pt1,pt2)

    if pt1=='A-plumipes' and pt2=='female':
        anthophora_plumipes_nan_class.append(i)
    elif pt1=='B-terrestris' and pt2=='queen':
        bombus_terrestris_queen_class.append(i)
    elif pt1=='B-terrestris' and pt2=='worker':
        bombus_terrestris_worker_class.append(i)
    elif pt1=='B-pratorum' and pt2=='queen':
        bombus_pratorum_queen_class.append(i)
    elif pt1=='B-pratorum' and pt2=='worker':
        bombus_pratorum_worker_class.append(i)
    elif pt1=='B-hypnorum' and pt2=='queen':
        bombus_hypnorum_queen_class.append(i)
    elif pt1=='B-hypnorum' and pt2=='worker':
        bombus_hypnorum_worker_class.append(i)
    elif pt1=='B-pascuorum' and pt2=='queen':
        bombus_pascuorum_queen_class.append(i)

# bombus_terrestris_queen_class=[]
# for i in wav_files:
#     g=i.split('_')
#     pt1=g[2]
#     pt2=g[3]

#     if pt1=='B-terrestris' and pt2=='queen':
#         bombus_terrestris_queen_class.append(i)

# bombus_terrestris_worker_class=[]
# for i in wav_files:
#     g=i.split('_')
#     pt1=g[2]
#     pt2=g[3]

#     if pt1=='B-terrestris' and pt2=='worker':
#         bombus_terrestris_worker_class.append(i)

# bombus_pratorum_queen_class=[]
# for i in wav_files:
#     g=i.split('_')
#     pt1=g[2]
#     pt2=g[3]

#     if pt1=='B-pratorum' and pt2=='queen':
#         bombus_pratorum_queen_class.append(i)

# bombus_pratorum_worker_class=[]
# for i in wav_files:
#     g=i.split('_')
#     pt1=g[2]
#     pt2=g[3]

#     if pt1=='B-pratorum' and pt2=='worker':
#         bombus_pratorum_worker_class.append(i)

# bombus_hypnorum_queen_class=[]
# for i in wav_files:
#     g=i.split('_')
#     pt1=g[2]
#     pt2=g[3]

#     if pt1=='B-hypnorum' and pt2=='queen':
#         bombus_hypnorum_queen_class.append(i)

# bombus_hypnorum_worker_class=[]
# for i in wav_files:
#     g=i.split('_')
#     pt1=g[2]
#     pt2=g[3]

#     if pt1=='B-hypnorum' and pt2=='worker':
#         bombus_hypnorum_worker_class.append(i)

# bombus_pascuorum_queen_class=[]
# for i in wav_files:
#     g=i.split('_')
#     pt1=g[2]
#     pt2=g[3]

#     if pt1=='B-hypnorum' and pt2=='worker':
#         bombus_pascuorum_queen_class.append(i)

print(len(anthophora_plumipes_nan_class))
print(len(bombus_terrestris_queen_class))
print(len(bombus_terrestris_worker_class))
print(len(bombus_hypnorum_queen_class))
print(len(bombus_hypnorum_worker_class))
print(len(bombus_pratorum_queen_class))
print(len(bombus_pratorum_worker_class))
print(len(bombus_pascuorum_queen_class))

# print(anthophora_plumipes_nan_class)
# x=[i for i in range(10)]
# x=torch.Tensor(x)
# print(torch.permute(x,-1))

from random import sample
# check out np.random.permutation(), don't forget to set a seed. 
# anthophora_plumipes_nan_class=sample(anthophora_plumipes_nan_class,len(anthophora_plumipes_nan_class))


def splits(my_class):
    Test=[]
    Train=[]
    Val=[]
    classlen=len(my_class)
    my_class=sample(my_class,classlen) #randomize the list
    # print(my_class)
    trainlen=round(0.7*classlen)
    testlen=round(0.10*classlen)
    vallen=(classlen-(testlen+trainlen))
    Train.extend(my_class[:trainlen])
    # print(Train)
    Test.extend(my_class[trainlen:trainlen+testlen])
    # print(Test)
    Val.extend(my_class[trainlen+testlen:])
    #print(len(Val))
    return Train, Test, Val
    # my_class.remove(:trainlen)
    # print(classlen,trainlen,testlen,vallen)


anthophora_plumipes_nan_class_train, anthophora_plumipes_nan_class_test, anthophora_plumipes_nan_class_val = splits(anthophora_plumipes_nan_class)
bombus_terrestris_queen_class_train, bombus_terrestris_queen_class_test, bombus_terrestris_queen_class_val = splits(bombus_terrestris_queen_class)
bombus_terrestris_worker_class_train, bombus_terrestris_worker_class_test, bombus_terrestris_worker_class_val = splits(bombus_terrestris_worker_class)
bombus_hypnorum_queen_class_train, bombus_hypnorum_queen_class_test, bombus_hypnorum_queen_class_val = splits(bombus_hypnorum_queen_class)
bombus_hypnorum_worker_class_train, bombus_hypnorum_worker_class_test, bombus_hypnorum_worker_class_val = splits(bombus_hypnorum_worker_class)
bombus_pratorum_queen_class_train, bombus_pratorum_queen_class_test, bombus_pratorum_queen_class_val = splits(bombus_pratorum_queen_class)
bombus_pratorum_worker_class_train, bombus_pratorum_worker_class_test, bombus_pratorum_worker_class_val = splits(bombus_pratorum_worker_class)
bombus_pascuorum_queen_class_train, bombus_pascuorum_queen_class_test, bombus_pascuorum_queen_class_val = splits(bombus_pascuorum_queen_class)

# TODO: stick lists together
# Write a for loop over train and add a backslash \n to the end of each entry of train 

train = anthophora_plumipes_nan_class_train + bombus_terrestris_queen_class_train + bombus_terrestris_worker_class_train + bombus_hypnorum_queen_class_train + bombus_hypnorum_worker_class_train + bombus_pratorum_queen_class_train + bombus_pratorum_worker_class_train + bombus_pascuorum_queen_class_train
test = anthophora_plumipes_nan_class_test + bombus_terrestris_queen_class_test + bombus_terrestris_worker_class_test + bombus_hypnorum_queen_class_test + bombus_hypnorum_worker_class_test + bombus_pratorum_queen_class_test + bombus_pratorum_worker_class_test + bombus_pascuorum_queen_class_test 
val = anthophora_plumipes_nan_class_val + bombus_terrestris_queen_class_val + bombus_terrestris_worker_class_val + bombus_hypnorum_queen_class_val + bombus_hypnorum_worker_class_val + bombus_pratorum_queen_class_val + bombus_pratorum_worker_class_test + bombus_pascuorum_queen_class_val

# print(train)

Save_Path=r'C:\Users\alixa\OneDrive\Desktop\Prototype_Data'
f = open(Save_Path + "\Train.txt", "w") 

for i in train:
    f.write(i + '\n')

# f.writelines(train)
f.close()

f = open(Save_Path + "\Test.txt", "w") 

for i in test:
    f.write(i + '\n')
    
f.close()

f = open(Save_Path + "\Val.txt", "w") 

for i in val:
    f.write(i + '\n')
    
f.close()

#  r'C:\Users\alixa\OneDrive\Desktop\Prototype_Data'





