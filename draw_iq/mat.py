from scipy.io import loadmat
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import stft, windows

'''
TODO
1.把时间标尺改成一个连续变化的 ❌
2.确定一下持续时间和采样率，对于全频谱的检测可以考虑对全频谱切片切成6片按顺序送到模型里判断 ⭕ 持续时间和采样率要可以还原出原信号
3.确定一下颜色标尺用一个固定变化率的颜色标尺
4.确定一下用不用软件增益 ⭕ 用
5.画图时的命名方式确定一下 ⭕
6.重采样 ❌不需要由于信号的特征能否映射到图片上取决于采集设备的采样带宽是否够

论文研究内容:
7.信噪比作为颜色标尺
8.AWG信噪比加噪声做对比实验
'''

'''
采样率*时间长度为每次从IQ数据中计算拿的点数
采样率和时间长度一定要固定，如果采样率改变，时间长度也要变，要使每次拿的采样点数都差不多
要改图片的尺度的话只能改stft点数
但信号在时间和频率上的变化特征很大程度上取决于采样设备的采样带宽，画图时保持这些尺度不变，能还原出信号即可
'''

#
time_duration = 0.03  # 预设时间窗的长度(关键参数)，决定x尺度
fs = 100e6  # 采样带宽要和采样设备匹配,实验室设备的采样带宽一般为15MHZ，原论文中的最大采样带宽为100MHZ，采样带宽决定y轴尺度
slice_point = int(fs*time_duration)

fly_name = 'AVATA'  # 预设无人机的名称
sourceFolderPath = 'E:/360MoveData/Users/sam826001/Desktop/AVATA'  # 预设读取的目录
targetFolderPath = 'E:/360MoveData/Users/sam826001/Desktop/temp/'  # 预设保存的目录

stft_point = 2048  # 采样点数是每次y轴用的点数,1024就够用了


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
                augmentation_Zxx1 = 20*np.log10(np.abs(Zxx_I1))  # 是否选择增强
                # 画第一组IQ
                plt.figure()
                plt.pcolormesh([i*t_I1, (i+1)*t_I1], f_I1, augmentation_Zxx1)  # 画图时在颜色标尺上应用了一个软件增益使信号在图中更加明显，其实不用也行，用了还会丢失距离信息
                # plt.colorbar()  # 在做实验观察信号的时候可以加一个颜色标尺，做数据集时不用加颜色标尺
                plt.title(fly_name + (str(i)))
                plt.savefig(targetFolderPath + fly_name + str(i) + '2.4GHZ.jpg')
                plt.close()

                # 再画第二个通道的
                f_I2, t_I2, Zxx_I2 = stft(data_I2[i*slice_point: (i+1) * slice_point],
                                 fs, window=windows.hamming(stft_point), nperseg=stft_point)
                augmentation_Zxx2 = 20*np.log10(np.abs(Zxx_I2))  # 是否选择增强
                # 画第二组IQ
                plt.figure()
                plt.pcolormesh(t_I2, f_I2, augmentation_Zxx2)  # 画图时在颜色标尺上应用了一个软件增益使信号在图中更加明显，其实不用也行，用了还会丢失距离信息
                plt.colorbar()  # 在做实验观察信号的时候可以加一个颜色标尺，做数据集时不用加颜色标尺
                plt.title(fly_name + (str(i)))
                plt.savefig(targetFolderPath + fly_name + str(i) + '5.8GHZ.jpg')
                plt.close()

                i += 1


if __name__ == '__main__':
    main()