import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        """
        LoRA层：低秩适配层，用于减少参数和计算。
        Args:
            in_features (int): 输入特征的维度。
            out_features (int): 输出特征的维度。
            rank (int): 低秩矩阵的秩。
        """
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        
        # 低秩矩阵W_A（in_features x rank）和W_B（rank x out_features）
        self.W_A = nn.Parameter(torch.randn(in_features, rank))
        self.W_B = nn.Parameter(torch.randn(rank, out_features))
        
        # 偏置项（可选，可以学习）
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # 低秩近似计算
        return F.linear(x, torch.mm(self.W_A, self.W_B), self.bias)
