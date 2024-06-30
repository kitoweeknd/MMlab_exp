_base_ = [
    'D:/ML_Project/mmDLtoolbox/Drone_Thesis/res_co_dino/source_CFG.py',
]
# 修改数据集相关配置
data_root = 'E:/深度学习记录存储/数据集/signals_COCO/'
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

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/_annotations.coco.json',
        data_prefix=dict(img='val/')))

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'val/_annotations.coco.json')


test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=data_root + 'test/_annotations.coco.json',
        data_prefix=dict(img='test/')))
test_evaluator = dict(
    ann_file=data_root + 'test/_annotations.coco.json',
    outfile_prefix='./work_dirs/signals_test/test')

load_form = 'E:/深度学习记录存储/pretrain/swin_large_patch4_window12_384_22k.pth'