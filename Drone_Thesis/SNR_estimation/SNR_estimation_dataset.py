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


def create_dataset(data, snr):
    with h5py.File(data_path + drone_name + '_SNR_' + str(snr) + 'dB.h5', 'w') as f:
        f.create_dataset('I', data=data.real)
        f.create_dataset('Q', data=data.imag)


def show_spectrum(data, name):
    f, t, Zxx = stft(data[0: slice_point],
                     fs, window=windows.hamming(stft_points), nperseg=stft_points, noverlap=stft_points//2)
    plt.figure()
    aug = 20*np.log10(np.abs(Zxx))
    norm = np.abs(Zxx)
    plt.pcolormesh(t, f, aug if Zxx_aug else norm, cmap='jet')
    plt.title(name)
    plt.axis('off')
    if save:
        plt.savefig(path_figsave + str(name))
    plt.show()


def SNR_estimation(noise, signal):
    signal_power = np.mean(np.abs(signal.real) ** 2)
    noise_power = np.mean(np.abs(noise.real) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def data_reader(path):
    data = h5py.File(path, 'r')
    RF0_I = data['RF0_I'][0]
    RF0_Q = data['RF0_Q'][0]
    signal_c0 = RF0_I + 1j*RF0_Q
    RF1_I = data['RF1_I'][0]
    RF1_Q = data['RF1_Q'][0]
    signal_c1 = RF1_I + 1j*RF1_Q
    return signal_c0, signal_c1


def noisy(signal, snr_dB):
    signal_power = np.mean(np.abs(signal)**2)
    # Calculate current noise power
    noise_power = signal_power / (10 ** (snr_dB / 10))
    # Generate AWGN
    noise = np.sqrt(noise_power) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    # Add noise to the signal
    noisy_signal = signal + noise
    return noisy_signal
    # Add noise to the signal


def main():
    signal_c0, signal_c1 = data_reader(path_signal)
    noise_c0, noise_c1 = data_reader(path_noise)
    if CSNR:
        snr = SNR_estimation(noise_c0, signal_c0)
        print(snr)

    show_spectrum(signal_c0.real, 'signal0')
    show_spectrum(signal_c1.real, 'signal1')
    # show_spectrum(noise_c0.real, 'noise0')
    # show_spectrum(noise_c1.real, 'noise1')
    for i in range(-20, 25, 5):
        noisy_signal = noisy(signal_c0, i)
        # noisy_noise = noisy(noise_c1, i)
        show_spectrum(noisy_signal.real, i)
        create_dataset(noisy_signal, i)


if __name__ == '__main__':
    main()