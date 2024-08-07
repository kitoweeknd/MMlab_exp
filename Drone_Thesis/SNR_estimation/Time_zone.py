import h5py
from scipy.signal import stft, windows
import numpy as np
import matplotlib.pyplot as plt


fs = 100e6
stft_points = 2048
duration_time = 0.1
slice_point = int(fs*duration_time)
CSNR = True
Zxx_aug = True

path_figsave = 'E:/RFDroneVision/Air2S/pic/'
path_noise = 'E:/Drone_dataset/RFA/DroneRFa/background/T0000_D00_S0000.mat'
path_signal = 'E:/Drone_dataset/RFA/DroneRFa/Air2S/low/done_jet/T0101_D10_S0001.mat'
data_path = 'E:/RFDroneVision/Air2S/'
drone_name = 'Air2S'
save = False


def data_reader(path):
    data = h5py.File(path, 'r')
    RF0_I = data['RF0_I'][0]
    RF0_Q = data['RF0_Q'][0]
    signal_c0 = RF0_I + 1j*RF0_Q
    RF1_I = data['RF1_I'][0]
    RF1_Q = data['RF1_Q'][0]
    signal_c1 = RF1_I + 1j*RF1_Q
    return signal_c0, signal_c1



def main():
    signal_c0, signal_c1 = data_reader(path_signal)
    # 绘制时域波形图
    plt.figure(figsize=(10, 4))
    plt.plot([i for i in range(375000)], np.real(signal_c0[:375000]), label='I (Real)')
    plt.title('Time Domain Signal (IQ) C1')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot([i for i in range(375000)], np.imag(signal_c0[:375000]), label='Q (Imaginary)')
    plt.title('Time Domain Signal (Q) C1')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot([i for i in range(375000)], np.real(signal_c1[:375000]), label='I (Real)')
    plt.title('Time Domain Signal (IQ) C1')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot([i for i in range(375000)], np.imag(signal_c1[:375000]), label='Q (Imaginary)')
    plt.title('Time Domain Signal (Q) C2')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()



    print('Done')


if __name__ == '__main__':
    main()