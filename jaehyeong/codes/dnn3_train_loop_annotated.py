
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ✅ 1. DNN(3) 모델 정의
# 이 모델은 은닉층 구조를 유동적으로 구성할 수 있으며,
# BatchNorm, Dropout, ReLU 등을 선택적으로 적용 가능하다.
# 또한 내부에 가중치 초기화 및 파라미터 수 계산 함수도 포함한다.
class DNN(nn.Module):
    def __init__(self, hidden_dims, num_classes, dropout_ratio,
                 apply_batchnorm, apply_dropout, apply_activation, set_super):
        # nn.Module 초기화를 반드시 수행해야 모델 기능이 제대로 작동함
        if set_super:
            super().__init__()

        # hidden_dims는 [입력, 은닉1, 은닉2, ..., 은닉N] 형태
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()  # 레이어를 담는 리스트

        for i in range(len(self.hidden_dims) - 1):
            # 완전 연결층 (Fully Connected Layer)
            self.layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))

            # 학습 안정화 및 속도 향상을 위한 정규화
            if apply_batchnorm:
                self.layers.append(nn.BatchNorm1d(self.hidden_dims[i + 1]))

            # 비선형성 도입을 위한 활성화 함수
            if apply_activation:
                self.layers.append(nn.ReLU())

            # 과적합 방지를 위한 Dropout
            if apply_dropout:
                self.layers.append(nn.Dropout(dropout_ratio))

        # 출력층: 마지막 은닉층 → 클래스 수
        self.classifier = nn.Linear(self.hidden_dims[-1], num_classes)
        # 로그 소프트맥스: 분류 확률을 log 형태로 출력
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # 입력 이미지를 평탄화: [B, 1, 28, 28] → [B, 784]
        x = x.view(x.shape[0], -1)
        # 은닉층 통과
        for layer in self.layers:
            x = layer(x)
        # 출력층 통과
        x = self.classifier(x)
        # log 확률 반환
        return self.softmax(x)

    def weight_initialization(self, weight_init_method):
        # 가중치 초기화 함수: 원하는 방식으로 가중치 설정
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if weight_init_method == 'gaussian':
                    nn.init.normal_(m.weight)  # 평균 0, 표준편차 1
                elif weight_init_method == 'xavier':
                    nn.init.xavier_normal_(m.weight)  # 입력/출력 균형
                elif weight_init_method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight)  # ReLU 최적
                elif weight_init_method == 'zeros':
                    nn.init.zeros_(m.weight)  # 0으로 초기화
                nn.init.zeros_(m.bias)  # 바이어스는 항상 0

    def count_parameters(self):
        # 학습 가능한 파라미터 수 출력 함수
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ✅ 2. 데이터 로딩 (MNIST)
# 손글씨 숫자 데이터셋을 PyTorch에서 쉽게 불러올 수 있도록 제공
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ✅ 3. 모델 생성 및 초기화
# 은닉층 구조 정의
hidden_dims = [784, 512, 256, 128]
model = DNN(hidden_dims, num_classes=10,
            dropout_ratio=0.2,
            apply_batchnorm=True,
            apply_dropout=True,
            apply_activation=True,
            set_super=True)

# 가중치 초기화: kaiming 방식은 ReLU와 잘 어울림
model.weight_initialization('kaiming')

# ✅ 4. 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Adam 옵티마이저는 학습률 자동 조절 기능이 있음
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# NLLLoss는 LogSoftmax와 함께 사용할 때 적합한 손실 함수
criterion = nn.NLLLoss()

# ✅ 5. 학습 루프
# 에폭(epoch) 수 설정
num_epochs = 1
for epoch in range(num_epochs):
    model.train()  # 학습 모드 켜기
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()        # 이전 gradient 초기화
        outputs = model(images)      # 예측값 계산
        loss = criterion(outputs, labels)  # 손실 계산
        loss.backward()              # 역전파
        optimizer.step()             # 파라미터 업데이트

        running_loss += loss.item()  # 누적 손실값 저장

    # 평균 손실 출력
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
