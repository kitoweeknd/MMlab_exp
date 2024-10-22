import h5py
from scipy.signal import stft, windows
import numpy as np
import matplotlib.pyplot as plt


fs = 100e6
stft_points = 2048
duration_time = 0.1
slice_point = int(fs*duration_time)

path_figsave = 'E:/360MoveData/Users/sam826001/Desktop/研1/无人机/无人机论文/实验部分数据/原始数据集中的部分/SNRestimation/fig_res/'

path_noise = 'E:/360MoveData/Users/sam826001/Desktop/研1/无人机/无人机论文/实验部分数据/原始数据集中的部分/SNRestimation/background/T0000_D00_S11111.mat'
path_signal = 'E:/360MoveData/Users/sam826001/Desktop/研1/无人机/无人机论文/实验部分数据/原始数据集中的部分/SNRestimation/Phantom3/T0001_D10_S0000.mat'


def show_spectrum(data, name):
    f, t, Zxx = stft(data[0: slice_point],
                     fs, window=windows.hamming(stft_points), nperseg=stft_points, noverlap=stft_points//2)
    plt.figure()
    aug = 20*np.log10(np.abs(Zxx))
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.abs(Zxx).max())
    plt.title(name)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.savefig(path_figsave + 'spectrum.png')
    plt.show()


def data_reader(path):
    data = h5py.File(path, 'r')
    RF0_I = data['RF0_I'][0]
    RF0_Q = data['RF0_Q'][0]
    RF1_I = data['RF1_I'][0]
    RF1_Q = data['RF1_Q'][0]
    return RF0_I, RF0_Q, RF1_I, RF1_Q


def main():
    signal_data_I0, signal_data_Q0, signal_data_I1, signal_data_Q1 = data_reader(path_signal)
    noise_data_I0, noise_data_Q0, noise_data_I1, noise_data_Q1 = data_reader(path_noise)
    show_spectrum(signal_data_I0, 'signal0')
    show_spectrum(signal_data_I1, 'signal1')
    show_spectrum(noise_data_I1, 'noise0')
    show_spectrum(noise_data_I0, 'noise1')


if __name__ == '__main__':
    main()