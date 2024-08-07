_base_ = 'C:/ML/MMlab_exp/configs/yolox/yolox_x_8xb8-300e_coco.py'


model = dict(
        bbox_head=dict(num_classes=15)
)

# 修改数据集相关配置
data_root = 'E:/数据集历史数据/drone_thesis_detection/MMlab/drone_coco_3Aug_DFresolition_TVT/'
metainfo = {
    'classes': ('Image_Transmission_signal_LFST',
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

train_dataset = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/annotation_coco.json',
        data_prefix=dict(img='train/')
    )
)

train_dataloader = dict(
    batch_size=4,
    dataset=train_dataset,
)


val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/annotation_coco.json',
        data_prefix=dict(img='val/')
    )
)

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


load_from = 'E:/Pretrain/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'