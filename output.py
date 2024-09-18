import torch

from arrowanglenet import ArrowAngleNet

# 模型加载和初始化
model = ArrowAngleNet().to('cpu')
# 加载已经训练好的模型权重
model.load_state_dict(torch.load('arrow_angle_model.pth'))

# 设置为评估模式
model.eval()

# 创建一个随机的输入张量，形状与训练数据一致
dummy_input = torch.randn(1, 3, 20, 20).to('cpu')  # 1是 batch_size，3是通道数 (RGB 图像)

# 导出为 ONNX 格式
torch.onnx.export(
    model,
    dummy_input,
    "arrow_angle_model.onnx",  # 输出的文件名
    input_names=['input'],  # 输入的名称
    output_names=['sin_output', 'cos_output'],  # 输出的名称
    opset_version=11,  # ONNX 版本
)

print("模型已成功导出为 ONNX 格式！")
