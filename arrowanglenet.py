import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ArrowAngleNet(nn.Module):
    def __init__(self):
        super(ArrowAngleNet, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 输出 32x20x20
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输出 64x20x20
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 输出 128x10x10
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 输出 256x5x5
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # 输出 512x5x5
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)  # 输出 1024x5x5

        # 全连接层
        self.fc1 = nn.Linear(1024 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)  # 输出角度
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # 卷积和池化操作
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 输出 32x10x10

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 输出 64x5x5

        x = F.relu(self.conv3(x))
        # 保持特征图大小为 5x5

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        # 展平
        x = x.view(x.size(0), -1)  # 输出 1024x5x5 => 1024*5*5

        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # x = self.dropout(x)
        x = self.fc5(x)  # 直接输出角度
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = ArrowAngleNet().to(device)
