_base_ = 'C:/ML/MMlab_exp/configs/faster_rcnn/faster-rcnn_r50-caffe-dc5_ms-1x_coco.py'


model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=15))
)

# 修改数据集相关配置
data_root = 'E:/数据集历史数据/MMlab/drone signal detect few shot.v3i.coco-mmdetection/'
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
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/')))
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
        ann_file=data_root + 'test/annotation_coco.json',
        data_prefix=dict(img='test/')))
test_evaluator = dict(
    ann_file=data_root + 'test/_annotations.coco.json',
    outfile_prefix='./work_dirs/signals_test/test')

# 用r50的即可
load_from = 'E:/pretrain/faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth'