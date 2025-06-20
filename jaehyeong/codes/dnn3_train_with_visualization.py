
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ✅ 1. DNN(3) 모델 정의
class DNN(nn.Module):
    def __init__(self, hidden_dims, num_classes, dropout_ratio,
                 apply_batchnorm, apply_dropout, apply_activation, set_super):
        if set_super:
            super().__init__()

        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()

        for i in range(len(self.hidden_dims) - 1):
            self.layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))

            if apply_batchnorm:
                self.layers.append(nn.BatchNorm1d(self.hidden_dims[i + 1]))

            if apply_activation:
                self.layers.append(nn.ReLU())

            if apply_dropout:
                self.layers.append(nn.Dropout(dropout_ratio))

        self.classifier = nn.Linear(self.hidden_dims[-1], num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return self.softmax(x)

    def weight_initialization(self, weight_init_method):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if weight_init_method == 'gaussian':
                    nn.init.normal_(m.weight)
                elif weight_init_method == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                elif weight_init_method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight)
                elif weight_init_method == 'zeros':
                    nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ✅ 2. 데이터 로딩
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ✅ 3. 모델 생성 및 초기화
hidden_dims = [784, 512, 256, 128]
model = DNN(hidden_dims, num_classes=10, dropout_ratio=0.2,
            apply_batchnorm=True, apply_dropout=True, apply_activation=True, set_super=True)
model.weight_initialization('kaiming')

device = torch.device("cpu")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

# ✅ 4. 학습 + 시각화용 기록 리스트
num_epochs = 3
train_losses = []
test_accuracies = []

# ✅ 5. 학습 루프
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # ✅ 테스트 정확도 측정
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# ✅ 6. 손실 & 정확도 그래프
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label="Test Accuracy", color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
torch.cuda.empty_cache()

# ✅ 7. 예측 결과 이미지 10장 출력
model.eval()
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
example_data, example_targets = example_data.to(device), example_targets.to(device)

with torch.no_grad():
    output = model(example_data)

pred = output.argmax(dim=1, keepdim=True)

# 시각화
plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(example_data[i].cpu().squeeze(), cmap='gray')
    plt.title(f"GT: {example_targets[i].item()} / Pred: {pred[i].item()}")
    plt.axis("off")
plt.tight_layout()
plt.show()
torch.cuda.empty_cache()

# ✅ 2. 혼동 행렬 (Confusion Matrix)
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)])
disp.plot(cmap="Blues", xticks_rotation="horizontal")
plt.title("📊 Confusion Matrix")
plt.grid(False)
plt.show()
torch.cuda.empty_cache()

# ✅ 3. 클래스별 정확도 출력
class_correct = [0] * 10
class_total = [0] * 10

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        for label, pred in zip(labels, preds):
            class_total[label.item()] += 1
            if label == pred:
                class_correct[label.item()] += 1

print("\n🎯 클래스별 정확도:")
for i in range(10):
    acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print(f"  숫자 {i}: {acc:.2f}% 정확도")