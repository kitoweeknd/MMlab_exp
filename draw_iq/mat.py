from scipy.io import loadmat
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import stft, windows

'''
TODO
1.把时间标尺改成一个连续变化的
2.确定一下持续时间和采样率，对于全频谱的检测可以考虑对全频谱切片切成6片按顺序送到模型里判断
3.确定一下颜色标尺用一个固定变化率的颜色标尺
4.确定一下用不用软件增益
5.画图时的命名方式确定一下

持续时间要考虑信号在图中的长度
采样率决定了信号在图中的数量和宽度，采样率还要考虑设备的性能，我们实验室的USRP设备的采样率一般为15MHZ
'''

time_duration = 0.08  # 预设时间窗的长度(关键参数)
fs = 100e6  # 采样率固定,实验室设备的采样率一般为15MHZ，原论文中的最大采样率为100MHZ
slice_point = int(fs*time_duration)

fly_name = 'Air2s'  # 预设无人机的名称
sourceFolderPath = 'E:/360MoveData/Users/sam826001/Desktop/AVATA'  # 预设读取的目录
targetFolderPath = 'E:/360MoveData/Users/sam826001/Desktop/temp'  # 预设保存的目录

stft_point = 2048


def main():
    # 设置一个循环使程序可以连续的读取一个文件中的所有图片
    while True:
        re_files = os.listdir(sourceFolderPath)
        for file in re_files:
            filePath = os.path.join(sourceFolderPath, file)
            file_size = os.path.getsize(filePath)
            data = h5py.File(filePath, 'r')
            data_I1 = data['RF0_I'][0]
            # data_Q1 = data['RF0_Q'][0]  # 同一组信号的IQ读一路就行
            data_I2 = data['RF1_I'][0]
            # data_Q2 = data['RF1_I'][0]  # 同一组信号的IQ读一路就行
            i = 0
            while (i+1)*slice_point <= len(data_I1):
                # 先画第一个通道的
                f_I1, t_I1, Zxx_I1 = stft(data_I1[i*slice_point: (i+1) * slice_point],
                                 fs, window=windows.hamming(stft_point), nperseg=stft_point)
                plt.figure()
                plt.pcolormesh(t_I1, f_I1, 20*np.log10(np.abs(Zxx_I1)))  # 画图时在颜色标尺上应用了一个软件增益使信号在图中更加明显，其实不用也行，用了还会丢失距离信息
                plt.colorbar()  # 在做实验观察信号的时候可以加一个颜色标尺，但是将信号画出来的话不用加颜色标尺
                plt.title(fly_name.join('i'))
                plt.savefig(targetFolderPath)
                plt.close()
                # 再画第二个通道的
                f_I2, t_I2, Zxx_I2 = stft(data_I2[i*slice_point: (i+1) * slice_point],
                                 fs, window=windows.hamming(stft_point), nperseg=stft_point)
                plt.figure()
                plt.pcolormesh(t_I2, f_I2, 20*np.log10(np.abs(Zxx_I2)))  # 画图时在颜色标尺上应用了一个软件增益使信号在图中更加明显，其实不用也行，用了还会丢失距离信息
                plt.colorbar()  # 在做实验观察信号的时候可以加一个颜色标尺，但是将信号画出来的话不用加颜色标尺
                plt.title(fly_name.join('i'))
                plt.savefig(targetFolderPath)
                plt.close()

                i += 1


if __name__ == '__main__':
    main()