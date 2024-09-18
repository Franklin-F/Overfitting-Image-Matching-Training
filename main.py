import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from arrowanglenet import ArrowAngleNet
import torch.nn as nn
from arrowdataset import ArrowDataset

image_folder = './arrow_images'
dataset = ArrowDataset(image_folder)

# 使用 DataLoader 加载数据
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 初始化模型、损失函数和优化器
model = ArrowAngleNet()
criterion = nn.MSELoss()  # 均方误差
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.01)
num_epochs = 10000  # 你可以根据需要调整训练轮数

for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0

    for images, angles in train_loader:
        images, angles = images.to(device), angles.to(device).unsqueeze(1)  # 将数据移动到 GPU

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, angles)

        # 反向传播和优化
        optimizer.zero_grad()  # 清除梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        running_loss += loss.item()
        # scheduler.step()

    print(f'第 [{epoch + 1}/{num_epochs}] 轮，损失值: {running_loss / len(train_loader):.4f}')

print("训练完成！")

# 保存模型为 .pth 格式
torch.save(model, 'arrow_angle_model.pth')  # 保存整个模型
torch.save(model.state_dict(), 'arrow_angle_model_state_dict.pth')  # 保存模型的状态字典（推荐）

# 导出为 .onnx 格式
dummy_input = torch.randn(1, 3, 20, 20).to(device)  # 假设输入图像大小为 20x20
torch.onnx.export(model, dummy_input, "arrow_angle_model.onnx", opset_version=11)

print("模型已导出为 .pth 和 .onnx 格式")
