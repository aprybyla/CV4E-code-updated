import os
import pandas as pd

blank = "Blank"

base_path = "/mnt/c/Users/alixa/OneDrive/Desktop/Prototype_Data/"

test_df = pd.read_csv(base_path + "Clipped_Files_for_bc_Test.txt", sep=' ',
    header=None, names=["File_Name", "Presence_Absence"])

write_my_file = open(base_path + "Clipped_Files_for_mc_Test.txt", "w")

for index, row in test_df.iterrows():
    clas = None
    if row.Presence_Absence == 0:
        clas = blank
    elif "plumipes_female" in row.File_Name: 
        clas = "Anthophora-plumipes-nan"
    elif "terrestris_queen" in row.File_Name:
        clas = "Bombus-terrestris-queen"
    elif "terrestris_worker" in row.File_Name:
        clas = "Bombus-terrestris-worker"
    elif "pratorum_queen" in row.File_Name:
        clas = "Bombus-pratorum-queen"
    elif "pratorum_worker" in row.File_Name: 
        clas = "Bombus-pratorum-worker"
    elif "hypnorum_queen" in row.File_Name: 
        clas = "Bombus-hypnorum-queen" 
    elif "hypnorum_worker" in row.File_Name: 
        clas = "Bombus-hypnorum-worker"
    elif "pascuorum_queen" in row.File_Name:
        clas = "Bombus-pascuorum-queen"
        
    print(clas)
    write_my_file.write(row.File_Name + " " + clas)
    write_my_file.write('\n')