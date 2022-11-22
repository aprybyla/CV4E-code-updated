from asyncore import write
from queue import Full
from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram
from pathlib import Path
import csv
import ipdb

folder = "./Clipped_Audio_for_Binary_Classification_VAL/"
Path(folder).mkdir(exist_ok=True)

writing = open('/mnt/c/Users/alixa/OneDrive/Desktop/Prototype_Data/' + "Clipped_Files_for_bc_Val.txt", "w") 

f = open("Val.txt", "r")
for x in f: 
    
    audio_file = '/mnt/c/Users/alixa/OneDrive/Desktop/Prototype_Data/.wav_files/' + x[:-1]

    flight_times = open('/mnt/c/Users/alixa/OneDrive/Desktop/Prototype_Data/flight_times.csv')
    csvreader = csv.reader(flight_times)
    save_path = './'

    buzz_dictionary = {}

    master_clip_list = []
    master_label_list = []

    end_of_previous_segment = 0.0
    rows = []
    for k,row in enumerate(csvreader):
            if k==0:
                continue 
            rows.append(row)
            name = row[0]
            #print (name, audio_file)
            #ipdb.set_trace()
            if name == audio_file.split('/')[-1]:
                print (row[0], row[1], row[2], row[3])

                audio = Audio.from_file(audio_file)
                audio_segment = Audio.from_file(audio_file,offset=float(row[1]),duration=float(row[2])-float(row[1]))
                
                #clip_df = audio_segment.split_and_save(clip_duration=1.0,clip_overlap=0.5,final_clip="full",prefix=audio_file.split('/')[-1],destination='./Clipped_Audio_for_Binary_Classification')
                clips, clip_df = audio_segment.split(clip_duration=1.0,clip_overlap=0.5,final_clip="full")
                
                for i in range(0,len(clip_df)-1):
                    clip_name = audio_file.split('/')[-1] + "_" + str(end_of_previous_segment + clip_df.iloc[i].start_time) + '_' + str(end_of_previous_segment + clip_df.iloc[i].end_time)
                    clips[i].save(folder + clip_name +'.wav')
                    #ipdb.set_trace()

                    if row[3] == "TRUE" : 
                        writing.write(folder + clip_name + '.wav' + ' ' + "1" + "\n")
                    else : 
                        writing.write(folder + clip_name + '.wav' + ' ' + "0" + "\n")

                    #writing.write('./Clipped_Audio_for_Binary_Classification/' + clip_name +'.wav'+ ' ' + row[3] + "\n")
                
                end_of_previous_segment = float(row[2])
                    
writing.close()