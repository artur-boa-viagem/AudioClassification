import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from matplotlib import image as img
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models

possible_labels = ['Bird', 'Cat', 'Dog', 'Sheep', 'Cow', 'Monkey']
labels_id = dict((name, index) for index, name in enumerate(possible_labels))

def getSpectrogramFromImage(path_tensor):
    path = path_tensor.numpy().decode()  # Convert the path tensor to a string
    image = img.imread(path)
    # Convert RGBA to grayscale
    image_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])  # Using luminosity method
    return image_gray

path = ['../Spectrogram-DB/HOG/Cat/0.png', '../Spectrogram-DB/HOG/Cat/1.png', '../Spectrogram-DB/HOG/Cat/2.png']
raw_train_data = tf.data.Dataset.from_tensor_slices(path)
train_data = raw_train_data.map(
    map_func=lambda path: tf.py_function(func=getSpectrogramFromImage, inp=[path], Tout=tf.float32),
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)

model = models.Sequential([
    layers.Input(shape=(None, None, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])
