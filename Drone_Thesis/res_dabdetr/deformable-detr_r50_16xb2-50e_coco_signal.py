_base_ = 'C:\ML\MMlab_exp\configs\dab_detr\dab-detr_r50_8xb2-50e_coco.py'

# 修改数据集相关配置
data_root = 'E:/Dataset_log/drone_thesis_detection/MMlab/drone_coco_3Aug_DFresolition_TV/'
metainfo = {
    'classes': (
                'Image_Transmission_signal_LFST',
                'Image_Transmission_signal_LFVST',
                'Image_Transmission_signal_MFST',
                'Image_Transmission_signal_Square',
                'Image_Transmission_signal_VLFVST',
                'Image_Transmission_signal__P4PR',
                'Tarains_flight_control',
                'frequency_hopping_signal_LFMT',
                'frequency_hopping_signal_LFST',
                'frequency_hopping_signal_SFLT',
                'frequency_hopping_signal_SFMT',
                'frequency_hopping_signal_SFST',
                'frequency_hopping_signal_Square',
                'frequency_hopping_signal_VLFMT',
                'yunzhuo_flight_control2',
                ),
    'palette': [
        (220, 20, 60),
    ]
}
model = dict(bbox_head=dict(num_classes=15))
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/annotation_coco.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/annotation_coco.json',
        data_prefix=dict(img='val/')))

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'val/annotation_coco.json')
test_dataloader = val_dataloader
test_evaluator = val_evaluator


load_from = 'E:/Pretrain/dab-detr_r50_8xb2-50e_coco_20221122_120837-c1035c8c.pth'