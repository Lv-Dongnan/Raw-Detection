import torch

# 保存模型
def save_model(model, path):
    torch.save(model.state_dict(), path)

# 加载模型
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
