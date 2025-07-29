import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from ResNetWithLoRA import ResNetWithLoRA
from data_loader import load_raw_image

# 自定义数据集
class RawImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_raw_image(self.image_paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# 设置数据集和数据加载器
train_image_paths = ['path_to_raw_image1.raw', 'path_to_raw_image2.raw']  # 示例路径
train_labels = [0, 1]  # 示例标签
train_dataset = RawImageDataset(train_image_paths, train_labels)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# 定义模型、优化器和损失函数
model = ResNetWithLoRA(pretrained=True, rank=4)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练过程
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
