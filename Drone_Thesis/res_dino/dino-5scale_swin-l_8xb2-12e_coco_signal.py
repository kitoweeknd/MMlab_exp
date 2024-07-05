_base_ = 'C:/ML/MMlab_exp/configs/dino/dino-5scale_swin-l_8xb2-12e_coco.py'


# 修改数据集相关配置
data_root = 'E:/数据集历史数据/MMlab/drone_coco/'
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
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=data_root + 'test/annotation_coco.json',
        data_prefix=dict(img='test/')))
test_evaluator = dict(
    ann_file=data_root + 'test/annotation_coco.json',
    outfile_prefix='./work_dirs/signals_test/test')
# 中断训练



load_from = 'E:/pretrain/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth'