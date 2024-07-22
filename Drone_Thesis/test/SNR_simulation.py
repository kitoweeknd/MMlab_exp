import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft


def generate_frequency_hopping_signal(frequencies, hop_duration, sample_rate, total_duration):
    t = np.arange(0, total_duration, 1 / sample_rate)
    signal = np.zeros_like(t)

    num_hops = int(total_duration / hop_duration)
    for i in range(num_hops):
        start_idx = int(i * hop_duration * sample_rate)
        end_idx = int((i + 1) * hop_duration * sample_rate)
        freq = np.random.choice(frequencies)
        signal[start_idx:end_idx] = np.sin(2 * np.pi * freq * t[start_idx:end_idx])

    return t, signal


def add_awgn_noise(signal, snr):
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr / 10))
    noise = np.sqrt(noise_power) * np.random.normal(size=signal.shape)
    noisy_signal = signal + noise
    return noisy_signal


# 参数设置
frequencies = [100, 200, 300, 400, 500]  # 跳变频率集合
hop_duration = 0.1  # 每次频率跳变持续时间（秒）
total_duration = 5  # 信号总持续时间（秒）
sample_rate = 10000  # 采样率（Hz）
snr = -20  # 信噪比（dB）

# 生成跳频信号
t, hopping_signal = generate_frequency_hopping_signal(frequencies, hop_duration, sample_rate, total_duration)

# 添加AWGN噪声
noisy_hopping_signal = add_awgn_noise(hopping_signal, snr)


# 使用STFT绘制加噪后的跳频信号的时频图
f, t_stft, Zxx = stft(noisy_hopping_signal, fs=sample_rate, nperseg=256)
plt.figure(figsize=(12, 6))
plt.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud')
plt.title('STFT Magnitude of Noisy Frequency Hopping Signal')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Magnitude')
plt.tight_layout()
plt.show()
plt.savefig('-20')
