import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
# import torch.nn.init as init
from VGG16.VGG16_Cifar10_Model import VGG16_Cifar10_Model


vgg_model = VGG16_Cifar10_Model()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cifar10_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar10_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(cifar10_train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(cifar10_test_dataset, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg_model.parameters(), lr=0.0001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg_model = vgg_model.to(device)

num_epochs = 200
for epoch in range(num_epochs):
    vgg_model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = vgg_model.forward_with_fc(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print(f'loss:{loss.item()}')

    # 可以在每个epoch结束后保存模型
    if epoch >= 199:
        torch.save(vgg_model.state_dict(), f'vgg16_cifar10_epoch_{epoch + 1}.pth')

    # 在测试集上评估模型
    vgg_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = vgg_model.forward_with_fc(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {accuracy}')

print("Training complete.")
