# 新配置继承了基本配置，并做了必要的修改
_base_ = 'C:/ML/MMlab_exp/configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_1x_coco.py'

# 我们还需要更改 head 中的 num_classes 以匹配数据集中的类别数
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=9)))

# 修改数据集相关配置
data_root = 'E:/数据集历史数据/MMlab/signal_coco/'
metainfo = {
    'classes': ('DJ_FlightCon',
                'DJ_PCTrans',
                'singal1',
                'singal2',
                'singal3',
                'singal4',
                'singal5',
                'singal6',
                'singal7',
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
        ann_file=data_root + 'test/annotation_coco.json',
        data_prefix=dict(img='test/')))
test_evaluator = dict(
    ann_file=data_root + 'test/annotation_coco.json',
    outfile_prefix='./work_dirs/signals_test/test')


# 使用预训练的 faster R-CNN 模型权重来做初始化，可以提高模型性能
load_from = 'E:/pretrain/faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth'
