import os
from math import floor
import numpy as np

baseDir = '../Spectrogram-DB/'

for dir in ['HOG/', 'LBP/']:
    typeDir = baseDir + dir 
    for folder in os.listdir(typeDir + 'Train/'):
        folderName = os.fsdecode(folder)
        files = np.array([os.fsdecode(file) for file in os.listdir(typeDir + 'Train/'+ folderName)])
        testFiles = np.random.choice(files, floor(len(files)*0.15), replace=False)
        for file in testFiles:
            os.system('mv ' + typeDir + 'Train/' + folderName + '/' + file + ' ' + typeDir + 'Test/' + folderName + '/' + file)
