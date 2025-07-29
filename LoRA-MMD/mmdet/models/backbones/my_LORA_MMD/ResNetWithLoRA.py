from torchvision.models import resnet50
from LoRALayer import LoRALayer
import torch.nn as nn

class ResNetWithLoRA(nn.Module):
    def __init__(self, pretrained=True, rank=8):
        super(ResNetWithLoRA, self).__init__()
        
        # 加载预训练的ResNet-50模型
        self.resnet = resnet50(pretrained=pretrained)
        
        # 替换ResNet的全连接层为LoRA层
        self.fc = LoRALayer(2048, 1000, rank)  # ResNet-50的输出维度是2048，分类任务的类别数为1000

        # 冻结ResNet的卷积层，只有LoRA层可训练
        for param in self.resnet.parameters():
            param.requires_grad = False  # 冻结ResNet的卷积层
        self.fc.requires_grad = True  # LoRA层是可训练的

    def forward(self, x):
        # 通过预训练的ResNet提取特征
        x = self.resnet(x)
        
        # 通过LoRA层进一步处理特征
        x = self.fc(x)
        
        return x
