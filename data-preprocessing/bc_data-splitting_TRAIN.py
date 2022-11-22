# Import packages to use in this code 
from asyncore import write
from queue import Full
from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram
from pathlib import Path
import csv
import ipdb

# Create a folder to store our clipped .wav files 
folder = "./Clipped_Audio_for_Binary_Classification_TRAIN/"
Path(folder).mkdir(exist_ok=True)

# Write and open a txt file where we will store the .wav file names and times of clipped data
writing = open('/mnt/c/Users/alixa/OneDrive/Desktop/Prototype_Data/' + "Clipped_Files_for_bc_Train.txt", "w") 

# Open the txt file where all of the names of the training dataset are stored. Begin a for loop. 
f = open("Train.txt", "r")
for x in f: 
    
    # Create variable audio_file (OpenSS variable) 
    audio_file = '/mnt/c/Users/alixa/OneDrive/Desktop/Prototype_Data/.wav_files/' + x[:-1]

    # Open and read the .csv file that associates the .wav file name, the flight times, and the presence/absence of a bee
    flight_times = open('/mnt/c/Users/alixa/OneDrive/Desktop/Prototype_Data/flight_times.csv')
    csvreader = csv.reader(flight_times)
    save_path = './'

    buzz_dictionary = {}

    # create a list for the sound clips, create a list for the associated labels (start and stop time, )
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

                #writing.write(name + ' ' + row[3] + '\n')
                #ipdb.set_trace()

                # open the audio file using this -> #audio = Audio.from_file(audio_file)
                # calling the segment() function with start and duration which is stored in row[1] and row[2]
                # using the segmented thing, make clips of __ durration using their clip function 
                # want to append the returned clips to master_clip_list and also the label (row[4]) to master_label_list



    #Using a dictionary, with the file name as the key, the values would be
    #this big list of lists where each list would be the audio start, the audio
    #stop, and the true/false value of bee presence 







    #audio = Audio.from_file(audio_file)

    #spectrogram = Spectrogram.from_audio(audio)
    #image = spectrogram.to_image(shape=(224,224),invert=True)

    #image.save(r'./first_image.png')
