import librosa as lb
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import feature

path_to_dataset = '../Animal-Sound-Dataset/'

def getCurrTime(sample, sr):
    return len(sample) / sr

def saveHOG(specgram, imagePath):
    # Compute HOG descriptor
    _, hog_image = feature.hog(specgram, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    plt.imshow(hog_image, cmap='gray')
    plt.axis('off')
    plt.savefig(imagePath, bbox_inches='tight', pad_inches=0)

def saveLBP(specgram, imagePath):
    specgram_db_rounded = np.round(specgram).astype(np.int64)
    lbp_image = feature.local_binary_pattern(specgram_db_rounded, P=8, R=1, method='uniform')
    plt.imshow(lbp_image, cmap='gray')
    plt.axis('off')
    plt.savefig(imagePath, bbox_inches='tight', pad_inches=0)

def main():
    maxTimeAudio = 5
    qtds = {}
    dir = os.fsencode(path_to_dataset)
    for sub_dir in os.listdir(dir):
        sub_dir_name = os.fsdecode(sub_dir)
        #the first element is the number of files with less than 5 seconds and the second is the number of files with more than 5 seconds
        qtds[sub_dir_name] = [0, 0]
        try:
            id = 0
            for file in os.listdir(path_to_dataset + sub_dir_name):
                filename = path_to_dataset + sub_dir_name + '/' + os.fsdecode(file)
                y, sr = lb.load(filename)
                wave = np.copy(y)
                print(filename)

                # Repeating the current wave to have exactly 5 seconds of duration
                while getCurrTime(wave, sr) < maxTimeAudio:
                    if getCurrTime(np.append(wave, y), sr) > maxTimeAudio:
                        fiveSecondsLength = maxTimeAudio*sr
                        toAppend = fiveSecondsLength - len(wave)
                        wave = np.append(wave, y[:toAppend])
                        break
                    wave = np.append(wave, y)
                
                # Compute the spectrogram
                specgram = lb.feature.melspectrogram(y=wave, sr=sr)

                # Convert to decibels
                specgram_db = lb.power_to_db(specgram, ref=np.max)

                baseDir = '../Spectrogram-DB/' 
                baseHOGDir = baseDir + 'HOG/' + sub_dir_name
                baseLBPDir = baseDir + 'LBP/' + sub_dir_name

                #debug
                # print(filename)
                # plt.figure(figsize=(10, 4))
                # lb.display.specshow(specgram_db, sr=sr, x_axis='time', y_axis='mel')
                # plt.colorbar(format='%+2.0f dB')
                # plt.title('Spectrogram')
                # plt.xlabel('Time (s)')
                # plt.ylabel('Frequency (Hz)')
                # plt.tight_layout()
                # plt.show()

                saveLBP(specgram_db, baseLBPDir + '/' + str(id) + '.png')
                saveHOG(specgram_db, baseHOGDir + '/' + str(id) + '.png')
                id += 1

        except NotADirectoryError:
            print('Not a directory')
            continue
    print(qtds)

if __name__ == "__main__":
    main()
