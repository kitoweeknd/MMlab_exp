_base_ = '../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

signal_classes = ['DJ_FlightCon', 'DJ_PCTrans', 'singal1', 'singal2',
                  'singal3', 'singal4', 'singal5', 'singal6', 'singal7']
datasets_type = 'CocoDataset'
path = 'data/signal/'

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        type=datasets_type,
        metainfo=dict(classes=signal_classes),
        data_root=path,
        ann_file='train/annotation_coco.json',
        data_prefix=dict(img='train/')
    )
)
val_dataloader = dict(
    dataset=dict(
        type=datasets_type,
        metainfo=dict(classes=signal_classes),
        data_root=path,
        ann_file='valid/annotation_coco.json',
        data_prefix=dict(img='valid/')
    )
)
test_dataloader = dict(
    dataset=dict(
        type=datasets_type,
        metainfo=dict(classes=signal_classes),
        data_root=path,
        ann_file='test/annotation_coco.json',
        data_prefix=dict(img='test/')
    )
)
val_evaluator = dict(  # 验证过程使用的评测器
    type='CocoMetric',  # 用于评估检测和实例分割的 AR、AP 和 mAP 的 coco 评价指标
    ann_file=path + 'valid/annotation_coco.json',  # 标注文件路径
    metric=['bbox'],  # 需要计算的评价指标，`bbox` 用于检测，`segm` 用于实例分割
    format_only=False)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=path + 'test/annotation_coco.json',
    metric=['bbox'],
    format_only=True,  # 只将模型输出转换为 coco 的 JSON 格式并保存
    outfile_prefix='./work_dirs/coco_detection/test')  # 要保存的 JSON 文件的前缀