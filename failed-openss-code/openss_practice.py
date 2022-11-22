from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram

import csv

audio_file = r'./.wav_files/01-04-2022_RBGE-022_B-hypnorum_queen_BEE-PRESENT_Audacity.wav.wav'
flight_times = open(r'./flight_times.csv')
csvreader = csv.reader(flight_times)

rows = []
for row in csvreader: 
        rows.append(row)
        name = './.wav_files/'+row[0]
        #print (name)
        if name == audio_file:
            print (row[0], row[1], row[2], row[3])





#audio = Audio.from_file(audio_file)

#spectrogram = Spectrogram.from_audio(audio)
#image = spectrogram.to_image(shape=(224,224),invert=True)

#image.save(r'./first_image.png')
