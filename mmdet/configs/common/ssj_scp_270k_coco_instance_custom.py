from mmengine.config import read_base

with read_base():
    from .ssj_270_coco_instance_custom import *

from mmdet.datasets import MultiImageMixDataset
from mmdet.datasets.transforms import CopyPaste

# dataset settings
dataset_type = CocoDataset
data_root = 'E:/数据集历史数据/drone_thesis_detection/MMlab/drone_coco_3Aug_DFresolition_TV/'
image_size = (1024, 1024)
backend_args = None
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


load_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True, with_mask=True),
    dict(
        type=RandomResize,
        scale=image_size,
        ratio_range=(0.8, 1.25),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=image_size),
]
train_pipeline = [
    dict(type=CopyPaste, max_num_pasted=100),
    dict(type=PackDetInputs)
]

train_dataloader.update(
    dict(
        type=MultiImageMixDataset,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file=data_root + 'annotation_coco.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=load_pipeline,
            backend_args=backend_args),
        pipeline=train_pipeline))
