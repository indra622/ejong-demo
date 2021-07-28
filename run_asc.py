import os
import sys

import tensorflow as tf

from src.asc import preprob, process, feats
from src.convert import csv_to_json

inputfilename = sys.argv[1]
inputfilepath = 'input/input-48k.wav'
filelist = 'input/asc-files.txt'
csvfilepath='output/asc-result.csv'
jsonfilepath = 'output/asc-result.json'

model_path = 'model/3model-183-0.3535-0.9388.tflite'


os.system('ffmpeg -loglevel 8 -y -i '+ inputfilename +' -acodec pcm_s16le -ac 1 -ar 48000 '+inputfilepath+' || exit 1;')
os.system('sox --i '+inputfilepath)

preprob(inputfilepath, 'asc-files/', False)

os.system('find asc-files/ -name \'*.wav\' | sort > '+filelist+' || exit 1;')

model = tf.lite.Interpreter(model_path=model_path)
model.allocate_tensors()

with open(filelist, 'r') as infile:
    with open(csvfilepath, 'w') as outfile:
        outfile.write('audio_url'+','+'timestamp' + ',' + 'score' + ',' + 'predict')
        outfile.write('\n')
        files = infile.readlines()
        
        for i in range(len(files)):
            wavpath = files[i].replace('\n','')
            logmel_data = feats(wavpath)
            i = int(wavpath.split('/')[-1].replace('.wav',''))
            threshold = 0.5
            process(i, threshold, logmel_data, outfile, model, wavpath)
            

csv_to_json(csvfilepath, jsonfilepath)