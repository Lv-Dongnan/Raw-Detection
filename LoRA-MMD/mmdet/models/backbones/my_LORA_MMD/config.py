# 配置文件
class Config:
    # 数据路径配置
    TRAIN_IMAGE_PATHS = ['path_to_train_image1.raw', 'path_to_train_image2.raw']
    TEST_IMAGE_PATHS = ['path_to_test_image1.raw', 'path_to_test_image2.raw']
    LABELS = [0, 1]  # 分类标签
    
    # 模型配置
    PRETRAINED = True
    RANK = 4  # LoRA的rank参数
    
    # 训练参数
    BATCH_SIZE = 2
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    OPTIMIZER = 'Adam'  # 可选择Adam或SGD

    # 保存路径
    MODEL_SAVE_PATH = 'saved_model.pth'
