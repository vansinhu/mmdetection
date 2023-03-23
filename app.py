# import cv2
# import numpy as np
import gradio as gr

# from mmdet.apis import inference_detector, init_detector

# 加载模型配置文件和检查点
config_file = 'mmdetection/projects/application/pcb/rtmdet_tiny-300e_coco.py'
checkpoint_file = 'work_dirs/rtmdet_tiny-300e_coco/epoch_300.pth'
# model = init_detector(config_file, checkpoint_file, device='cuda:0')
# model = init_detector(config_file, checkpoint_file, device='cuda:0')

model = None


def detect_objects(image):
    # 调用mmyolo模型进行物体检测
    # result = inference_detector(model, image)

    # 绘制检测结果
    # for bbox in result[0]:
    #     x1, y1, x2, y2 = map(int, bbox[:4])
    #     score = bbox[4]
    #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.putText(image, str(round(score, 2)), (x1, y1 - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 将结果返回
    return image


# 定义输入和输出界面
input_image = gr.inputs.Image(label='Upload an image')
output_image = gr.outputs.Image(label='Output image', type='numpy')

# 创建界面并启动应用
gr.Interface(
    detect_objects,
    inputs=input_image,
    outputs=output_image,
    capture_session=True).launch(server_name='0.0.0.0')
