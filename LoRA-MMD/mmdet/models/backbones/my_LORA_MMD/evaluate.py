# 评估模型
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

# 示例评估
test_image_paths = ['path_to_test_image1.raw', 'path_to_test_image2.raw']
test_labels = [0, 1]
test_dataset = RawImageDataset(test_image_paths, test_labels)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

evaluate_model(model, test_loader)
