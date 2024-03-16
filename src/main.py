import librosa as lb
import matplotlib.pyplot as plt
import numpy as np
import os

path_to_dataset = '../Animal-Sound-Dataset/'

def plot_wave(wave, sample_rate):
    maxTime = 5
    dt = 1/sample_rate           # Time resolution
    T = len(wave) / sample_rate  # Duration of the signal (seconds)
    t = np.arange(0, T, dt)      # Time vector

    # Signal in time domain
    plt.subplot(3, 1, 1)
    plt.plot(t, wave)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform')
    plt.grid(True)

    # Compute the FFT
    X = np.fft.fft(wave)
    freqs = np.fft.fftfreq(len(wave), dt)

    # Signal in frequency domain
    plt.subplot(3, 1, 2)
    # Plot only the positive frequencies
    plt.plot(freqs[:len(freqs)//2], np.abs(X)[:len(freqs)//2])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FFT of Wave')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.specgram(wave + 1e-3, Fs=sample_rate, NFFT=1024, noverlap=512, scale='dB')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram (FFT)')
    plt.colorbar(label='Magnitude (dB)')

    plt.tight_layout()
    plt.show()

def main():
    qtds = {}
    dir = os.fsencode(path_to_dataset)
    for sub_dir in os.listdir(dir):
        sub_dir_name = os.fsdecode(sub_dir)
        #the first element is the number of files with less than 5 seconds and the second is the number of files with more than 5 seconds
        qtds[sub_dir_name] = [0, 0]
        try:
            for file in os.listdir(path_to_dataset + sub_dir_name):
                filename = path_to_dataset + sub_dir_name + '/' + os.fsdecode(file)
                wave, sr = lb.load(filename)
                if len(wave)/sr < 5:
                    plot_wave(wave, sr)
                    qtds[sub_dir_name][0] += 1
                else:
                    qtds[sub_dir_name][1] += 1
                    os.remove(filename)
        except NotADirectoryError:
            print('Not a directory')
            continue
    print(qtds)

if __name__ == "__main__":
    main()
