
'''
# 改模型路径
# 改share路径
# 改res_path
# 改保存图片的路径
# 所有路径都不能有中文
'''

import matplotlib.pyplot as plt
from scipy.signal import stft, windows
import sys
import argparse
from pathlib import Path
import os
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import time
from utils.torch_utils import select_device
from utils.general import non_max_suppression
from models.common import DetectMultiBackend
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


# 预设参数中大江无人机采样率为15MHZ持续时间为0.03
# 老数据集中的无人机采样率为100MHZ持续时间为0.1
# 不断读取文件夹内文件并拼接数据
time_duration = 0.1  # 固定时间
fs = 100e6  # 设置一个固定的采样率
slice_point = int(fs * time_duration)
stft_point = 2048

directory = "E:/hands/share/"
res_path = 'E:/hands/res/'
fig_save_path = 'E:/hands/save_fig/'

# filePath = 'E:/Drone_done/2024.4.3/2024-03-22-11-37-45_Freq2461.000MHzFs15000kHz.iq'  # 输入接口


def read_and_execute_algorithm(file_path):
    # 在这里编写你的算法，这里只是一个示例
    with open(file_path, 'r') as file:
        content = file.read()
        # 这里假设执行的算法是将内容转换为大写
        result = content.upper()
    return result


def main():
    while True:
        # 检测是否存在txt文件
        files = [f for f in os.listdir(directory) if f.endswith('.txt')]
        if files:
            # 读取第一个txt文件的内容并执行算法
            temp = os.path.join(directory, files[0])
            with open(temp, 'r') as file:
                file_path = file.read()
            detect(file_path)
            os.remove(temp)

        # 将算法得到的结果写入新的txt文件
            print("算法执行完毕，并将结果写入 result.txt 文件。")
        else:
            print("未检测到txt文件，继续等待...")
            time.sleep(1)  # 每秒检测一次


def detect(file_path):
    with open(file_path, 'rb') as fp:
        read_data = np.fromfile(fp, dtype=np.int16)
        dataI = read_data[::2]
        if len(dataI) >= slice_point:
            # 绘制 dataI 的 stft 图像
            for ii in range(len(dataI) // slice_point):
                f, t, Zxx = stft(dataI[ii * slice_point:(ii + 1) * slice_point],
                                    fs, window=windows.hamming(stft_point), nperseg=stft_point)

                plt.figure()
                plt.pcolormesh(t, f, 20*np.log10(np.abs(Zxx)))  # 获取复数的绝对值
                plt.title("I " + str(ii))
                # 显示图像
                plt.savefig(fig_save_path + 'temp')
                plt.close()

                """"
                #这里的转格式算法要重新找一个
                fig = plt.gcf()  # 获取当前图形
                fig.canvas.draw()  # 绘制图形到画布
                # 获取图像数组并转换为OpenCV格式
                img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # 转换为RGB图像数据
                img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # 重塑为图像形状
                # 转换为OpenCV的BGR格式
                img_data_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)  # RGB转BGR
                """
                return inference()



def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='C:/ML/YOLO/v5/yolov5sr/yolov5-master/runs/train/exp39/weights/best.pt',
                        help='model.pt path(s)')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    print(opt)
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    cudnn.benchmark = True
    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=False, data=ROOT / 'data/coco128.yaml', fp16=False)  # load FP32 model
    if half:
        model.half()  # to FP16
    # Get names and colors
    names = model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    name_list = []  # 一次预测的结果
    with torch.no_grad():
        showimg = cv2.imread(fig_save_path + 'temp.png')
        cv2.imshow('test', showimg)
        cv2.waitKey(0)
        img = showimg
        img = letterbox(img, new_shape=opt.img_size)[0]
        original_img = img
        # Convert
        # BGR to RGB, to 3x416x416
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                   agnostic=opt.agnostic_nms)
        print(pred)
        # Process detections
        for i, det in enumerate(pred):
            res = {}
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                res[names[int(c)]] = int(n)
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], showimg.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = '%s %.2f' % (names[int(cls)], conf)
                    name_list.append(names[int(cls)])
                    print(label)
                    plot_one_box(
                        xyxy, showimg, label=label, color=colors[int(cls)], line_thickness=2)
        # Process detections
        if 'Air2S' in res:
            print('检测到Air2S')
            s = "检测到Air2S！"  # 替换为您的语音文本
        elif 'AVATA' in res:
            print('检测到AVATA')
            s = "检测到AVATA！"  # 替换为您的语音文本
        elif 'FrskyX20' in res:
            print('检测到FrskyX20')
            s = "检测到FrskyX20！"  # 替换为您的语音文本
        elif 'DJi' in res:
            print('检测到DJi')
            s = "find DJi！"  # 替换为您的语音文本'
        elif 'FrskyX20' in res:
            print('检测到Frsky')
            s = "检测到FrskyX20！"  # 替换为您的语音文本
        elif 'Fubuta' in res:
            print('检测到Fubuta')
            s = "检测到Fubuta！"  # 替换为您的语音文本
        elif 'Inspire2' in res:
            print('检测到Inspire2')
            s = "检测到Inspire2！"  # 替换为您的语音文本
        elif 'Matrice100' in res:
            print('检测到Matrice100')
            s = "检测到检测到Matrice100！"  # 替换为您的语音文本
        elif 'Matrice200' in res:
            print('检测到Matrice200')
            s = "检测到Matrice200！"  # 替换为您的语音文本
        elif 'Matrice600Pro' in res:
            print('检测到Matrice600Pro')
            s = "检测到检测到Matrice600Pro！"  # 替换为您的语音文本
        elif 'Phantom3' in res:
            print('检测到Phantom3')
            s = "检测到检测到Phantom3！"  # 替换为您的语音文本
        elif 'Phantom4PRO' in res:
            print('检测到Phantom4PRO')
            s = "检测到检测到Phantom4PRO！"  # 替换为您的语音文本
        elif 'PhantomPRORTK' in res:
            print('检测到Phantom4PRORTK')
            s = "检测到检测到Phantom4PRORTK！"  # 替换为您的语音文本
        elif 'Tarains_Plus' in res:
            print('Tarains_Plus')
            s = "检测到Tarains_Plus！"  # 替换为您的语音文本
        else:
            s = '未检测到无人机'

        with open(res_path + "result.txt", "w") as output_file:
            output_file.write(s)
        return 'Done'


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


if __name__ == '__main__':
    main()
