from mmdet.apis import DetInferencer


inferencer = DetInferencer(model='E:/训练结果历史数据/Drone_thesis/signal_detect/exp6_efficientNet/dino-5scale_swin-l_8xb2-12e_coco_signal.py',
                           weights='E:/训练结果历史数据/Drone_thesis/signal_detect/exp6_efficientNet/epoch_60.pth',
                           device='cuda:0')
inferencer('E:/检测结果历史数据/Drone_Thesis/',
           out_dir='E:/检测结果历史数据/Drone_Thesis/inference_res/',
           no_save_pred=False)
