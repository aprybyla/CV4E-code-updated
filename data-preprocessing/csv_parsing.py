####################################################################
# This code is to help Alix zip .wav and .txt files, to export the #
# contents of .txt files into .csv files as buzzes or not buzzes   #
# and the merge the written data like weight and pollen score to   #
# the acoustic data.                                               #
####################################################################                                      
# Code was written by Eli Cole and Justin Key at CalTech and       #
# Alixandra Prybyla at Uni of Edinburgh                            #
####################################################################

import os
import pandas as pd

base_path = r'C:\Users\alixa\OneDrive\Desktop\Prototype_Data'

wav_files = os.listdir(os.path.join(base_path, r'.wav_files'))

def parse_text_file(load_path):
    with open(load_path, 'r') as f:
        lines = f.readlines()
    start_list = []
    stop_list = []
    label_list = []
    for l in lines:
        l = l.rstrip()
        fields = l.split('\t')
        start_list.append(float(fields[0])) 
        stop_list.append(float(fields[1]))
        label_list.append(str(fields[2]))
    return start_list, stop_list, label_list

df=pd.read_csv(r'C:\Users\alixa\OneDrive\Desktop\Prototype_Data\Prototype_CSV.csv')
print(df.columns)

presence_list = []
file_name_list = []
start_list = []
stop_list = []
label_list = []
date_list = []
unique_specimen_number_list = []
species_list = []
caste_list = []
sex_list = []
pollen_list = []
mites_list = []
weight_list = []
itd_list = []
for fname in wav_files:
    fname_text = fname[:-8] + '.txt.txt'
    cur_start_list, cur_stop_list, cur_label_list = parse_text_file(os.path.join(base_path, r'.txt_files', fname_text))
    start_list.extend(cur_start_list)
    stop_list.extend(cur_stop_list)
    label_list.extend(cur_label_list)
    cur_file_name_list = [fname] * len(cur_label_list)
    file_name_list.extend(cur_file_name_list)
    cur_presence_list = ['Bee Flight' in label for label in cur_label_list]
    presence_list.extend(cur_presence_list)
    df_file = df[df['.wav file']==fname[:-4]]
    try:
        if  fname == '21-03-2022_RBGE-008_B-pratorum_worker_BEE-PRESENT_Auadcity.wav.wav':
            print(df_file.Date.values)
            print(df_file['Unique Specimen #'])
            print(df_file['Caste'])
            print(df_file['Sex'].values[0])
            print(df_file['Pollen '])
            print(df_file['# of Mites'])
            print(df_file['Weight (g)'])
            print(df_file['ITD (mm)'])
            print(df_file['Species'])
        cur_date_list = [df_file.Date.values[0]] * len(cur_label_list)
        cur_spec_list = [df_file['Unique Specimen #'].values[0]] * len(cur_label_list)
        cur_caste_list = [df_file['Caste'].values[0]] * len(cur_label_list)
        cur_sex_list = [df_file['Sex'].values[0]] * len(cur_label_list)
        cur_pollen_list = [df_file['Pollen '].values[0]] * len(cur_label_list)
        cur_mites_list = [df_file['# of Mites'].values[0]] * len(cur_label_list)
        cur_weight_list = [df_file['Weight (g)'].values[0]] * len(cur_label_list)
        cur_itd_list = [df_file['ITD (mm)'].values[0]] * len(cur_label_list)
        cur_species_list = [df_file['Species'].values[0]] * len(cur_label_list)
    except Exception as e:
        print('error', fname, e)
        cur_date_list = ['01/01/2001'] * len(cur_label_list)
        cur_spec_list = ['RBGE-999'] * len(cur_label_list)
        cur_caste_list = ['UNREAL'] * len(cur_label_list)
        cur_sex_list = ['UNREAL'] * len(cur_label_list)
        cur_pollen_list = ['UNREAL'] * len(cur_label_list)
        cur_mites_list = ['UNREAL'] * len(cur_label_list)
        cur_weight_list = ['UNREAL'] * len(cur_label_list)
        cur_itd_list = ['UNREAL'] * len(cur_label_list)
        cur_species_list = ['UNREAL'] * len(cur_label_list)

    date_list.extend(cur_date_list)
    unique_specimen_number_list.extend(cur_spec_list)
    caste_list.extend(cur_caste_list)
    sex_list.extend(cur_sex_list)
    pollen_list.extend(cur_pollen_list)
    mites_list.extend(cur_mites_list)
    weight_list.extend(cur_weight_list)
    itd_list.extend(cur_itd_list)
    species_list.extend(cur_species_list)

    #print(cur_date_list)

import csv

f = open(r'C:\Users\alixa\OneDrive\Desktop\Prototype_Data\flight_times.csv', 'w', newline='')
writer = csv.writer(f)
writer.writerow(["File Name", "Start", "Stop", "Presence", "Date", "Unique Specimen #", "Species", "Caste", "Sex", "Pollen", "Mites", "Weight (g)", "ITD (mm)"])
#
#for l in [file_name_list, start_list, stop_list, presence_list, date_list, unique_specimen_number_list, species_list, caste_list, sex_list, pollen_list, mites_list, weight_list, itd_list]:
    #print(len(l))

for file_name, start, stop, presence, date, unique_specimen, species, caste, sex, pollen, mites, weight, ITD in zip(file_name_list, start_list, stop_list, presence_list, date_list, unique_specimen_number_list, species_list, caste_list, sex_list, pollen_list, mites_list, weight_list, itd_list):
    writer.writerow([file_name, start, stop, presence, date, unique_specimen, species, caste, sex, pollen, mites, weight, ITD])


f.close()
    







