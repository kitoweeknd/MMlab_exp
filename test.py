from mmdet.apis import DetInferencer

# 初始化模型
inferencer = DetInferencer('rtmdet_tiny_8xb32-300e_coco')

# 推理示例图片
inferencer('demo/demo.jpg', show=True)
