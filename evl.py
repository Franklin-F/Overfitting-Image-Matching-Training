import os

import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# 加载 ONNX 模型
onnx_model_path = 'arrow_angle_model.onnx'
ort_session = ort.InferenceSession(onnx_model_path)

# 查看 ONNX 模型的输入名称
for input_name in ort_session.get_inputs():
    print(f"模型输入名称: {input_name.name}")


# 加载图片并进行预处理
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')

    # 定义预处理转换
    preprocess = transforms.Compose([
        transforms.Resize((20, 20)),  # 调整图片大小
        transforms.ToTensor(),  # 转为张量
    ])

    # 执行预处理，增加批次维度 [1, 3, 20, 20]
    img_tensor = preprocess(image).unsqueeze(0)

    return img_tensor.numpy()


# 加载并预处理一张箭头图片
for i in os.listdir('val_img'):
    image_path = os.path.join('val_img', i)
    input_image = preprocess_image(image_path)
    outputs = ort_session.run(None, {'input.1': input_image})
    print(f"模型预测角度: {outputs[0][0][0]}, 截图角度: {i}, 相差: {abs(outputs[0][0][0] - float(i.split('.')[0]))}")
