_base_ = 'D:\ML_Project\mmDLtoolbox\configs\detr\detr_r50_8xb2-500e_coco.py'


model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=9))
)

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
        metainfo=metainfo,
        ann_file=data_root + 'test/annotation_coco.json',
        data_prefix=dict(img='test/')))
test_evaluator = dict(
    ann_file=data_root + 'test/annotation_coco.json',
    outfile_prefix='./work_dirs/signals_test/test')


# 用普通r101即可
load_from = 'E:/深度学习记录存储/pretrain/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'