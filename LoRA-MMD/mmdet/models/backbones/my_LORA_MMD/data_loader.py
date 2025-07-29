import rawpy
import numpy as np
from torchvision import transforms
from PIL import Image

def load_raw_image(image_path):
    """
    加载RAW图像并进行预处理。
    Args:
        image_path (str): RAW图像文件路径
    Returns:
        Tensor: 预处理后的图像数据
    """
    # 使用rawpy加载RAW图像
    raw = rawpy.imread(image_path)
    rgb = raw.postprocess()

    # 转换为PIL图像并进行预处理
    image = Image.fromarray(rgb)
    
    # 图像转换为Tensor并做标准化处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return transform(image).unsqueeze(0)  # 添加batch维度

# 示例
image_path = 'path_to_raw_image.raw'
input_image = load_raw_image(image_path)
print(input_image.shape)  # 输出：torch.Size([1, 3, 224, 224])
