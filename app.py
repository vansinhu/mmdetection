import os

import cv2
# import numpy as np
import gradio as gr

from mmdet.apis import inference_detector, init_detector


from argparse import ArgumentParser

from mmengine.logging import print_log

from mmdet.apis import DetInferencer


# os.system('pip install openmim')
# os.system('mim install "mmcv>=2.0.0rc0"')

# os.system('pip install -v -e .')
# os.system(
#     'wget https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth -O rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'  # noqa E501
# )

# # 加载模型配置文件和检查点
config_file = 'configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'
checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')
# # model = init_detector(config_file, checkpoint_file, device='cuda:0')






inferencer = DetInferencer(
    model='configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py',
    weights='rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth',
    device='cpu'
    )

def detect_objects(image):
    # 调用mmyolo模型进行物体检测
    result = inferencer(image, return_vis=True)

    # 绘制检测结果
    # for bbox in result.pred_instances.bboxes.tolist():
    #     x1, y1, x2, y2 = map(int, bbox[:4])
    #     score = bbox[3]
    #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.putText(image, str(round(score, 2)), (x1, y1 - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 将结果返回
    return result['visualization'][0]

def main():
    # init_args, call_args = parse_args()
    # TODO: Video and Webcam are currently not supported and
    #  may consume too much memory if your input folder has a lot of images.
    #  We will be optimized later.
    # data = inferencer(**call_args)

    # if call_args['out_dir'] != '' and not (call_args['no_save_vis']
    #                                        and call_args['no_save_pred']):
    #     print_log(f'results have been saved at {call_args["out_dir"]}')



    # 定义输入和输出界面
    input_image = gr.inputs.Image(label='Upload an image')
    output_image = gr.outputs.Image(label='Output image', type='numpy')

    # 创建界面并启动应用
    gr.Interface(
        detect_objects,
        inputs=input_image,
        outputs=output_image,
        description="关注 openmmlab 公众号，回复 mmapp 手手教你 cpu场景下部署明星算法 RTMDet-tiny",
        capture_session=True).launch(server_name='0.0.0.0', server_port=7666)


if __name__ == '__main__':
    main()
