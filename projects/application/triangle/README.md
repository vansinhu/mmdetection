# Triangle

```shell
cd path/to/mmdetection
mkdir data & cd data
wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220610-mmpose/triangle_dataset/Triangle_140_Keypoint_Dataset.zip
unzip Triangle_140_Keypoint_Dataset.zip
```

```shell
cd path/to/mmdetection
python tools/train.py projects/application/triangle/faster-rcnn_r50_fpn_1x_coco.py
```

