from mmdet.apis import DetInferencer


inferencer = DetInferencer(model='C:/ML/MMlab_exp/Drone_Thesis/res_faster_RCNN_FPN/20240517_144742/20240517_234822/vis_data/config.py',
                           weights='E:/训练结果历史数据/Drone_thesis/exp1_faster-RCNN/epoch_60.pth',
                           device='cuda:0')
inferencer('C:/ML/MMlab_exp/Drone_Thesis/res_faster_RCNN_FPN/20240517_144742/inference_input/',
           out_dir='C:/ML/MMlab_exp/Drone_Thesis/res_faster_RCNN_FPN/20240517_144742/inference_res/',
           no_save_pred=False)
