import os
import numpy as np
from matplotlib import image as img

# Classes of the dataset
possible_labels = ['Bird', 'Cat', 'Dog', 'Sheep', 'Cow']
# Dictionary to map the labels to integers
labels_id = dict((name, index) for index, name in enumerate(possible_labels))

# Get the paths to the images of the dataset
# @fileType can be 'Train/' or 'Test/'
# @typeDir can be 'HOG/' or 'LBP/'
def getEveryFile(fileType = 'Train/', typeDir = 'HOG/'):
    baseDir = '../Spectrogram-DB/'
    files = []
    for label in possible_labels:
        for file in os.listdir(baseDir + typeDir + fileType + label):
            files.append(baseDir + typeDir + fileType + label + '/' + file)
    return files

# Load the image and return the grayscale image and its label
def getSpectrogramFromImage(path):
    image = img.imread(path)
    # Convert RGBA to grayscale
    image_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])  # Using luminosity method
    return image_gray, labels_id[path.split('/')[-2]]
