# Copyright (c) OpenMMLab. All rights reserved.

# Please refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta for more details. # noqa
# mmcv >= 2.0.1
# mmengine >= 0.8.0

from mmengine.config import read_base

with read_base():
    from .ssj_Thesis import *

from mmdet.datasets import MultiImageMixDataset
from mmdet.datasets.transforms import CopyPaste

# dataset settings
dataset_type = CocoDataset
data_root = 'E:/深度学习记录存储/数据集/signals_COCO/'
image_size = (1024, 1024)
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
# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

# Standard Scale Jittering (SSJ) resizes and crops an image
# with a resize range of 0.8 to 1.25 of the original image size.
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
            ann_file='train/_annotations.coco.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=load_pipeline,
            backend_args=backend_args),
        pipeline=train_pipeline))
