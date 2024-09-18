import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# 自定义数据集类
class ArrowDataset(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为 float32 张量
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # 从文件名中获取角度，并将其转换为 float32
        # 如果文件有A_开头，那么取A_后面的数字
        name = os.path.splitext(img_name)[0]
        int_angle = name.split("_")[-1] if "A_" in name else name
        angle = float(int_angle) # 0 为文件名，1 为文件后缀
        angle = torch.tensor(angle, dtype=torch.float32)  # 确保角度为 float32 类型

        return image, angle


if __name__ == '__main__':

    image_folder = './arrow_images'
    dataset = ArrowDataset(image_folder)
    # 使用 DataLoader 加载数据，并确保每次加载都使用 GPU 计算
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for images, angles in train_loader:
        print(images.size(), angles)

