# nn.Module 이란?
# PyTorch에서 딥러닝 모델은 nn.Module이라는 클래스를 상속받아 만들게 돼 있다.
# nn.Module은 PyTorch에서 모델의 뼈대 역할을 하는 클래스.


import torch.nn as nn

class DNN(nn.Module):  # nn.Module을 상속
    def __init__(self):
        super().__init__()

## nn.Module의 기능
## 모델 안에 있는 모든 레이어 들을 자동으로 추적해줌.
## .to(device)로 GPU올릴 수 있음
## .parameters()로 모든 파라미터 꺼내줌
## .train() / .eval() 같은 훈련 모드 설정도 자동으로 됨
## state_dict()로 저장/불러오기도 쉬워짐

######### nn.module에서 레이어란 ?  #############
# 레이어는 계싼을 수행하는 단겔를 의미함
# 딥러닝은 입력 데이터를 조금씩 바꿔가며 원하는 결과를 내는 '층(layer)'을 여러개 쌓아 만드는 모델

##### 따라서 모델을ㅇ 만들 땐 무조건 mm.Module을 상속해서 씀

super().__init__()

## 부모 클래스인 nn.Module의 초기화 함수(__init__)를 실행시키는 코드
## 즉, 부모 클래스의 기능을 자식 클래스에 활성화 시켜주는 것

class DNN(nn.Module):
    def __init__(self):
        super().__init__()  # nn.Module이 가진 기능들 사용할 준비
## 이 친구를 쓰지않게 되면, 모델이 제대로 도앚ㄱ하지않아서 PyTorch가 내 레이어들을 추적 못함

##### hidden_dim = 64
##### self.fc1 = nn.Linear(784, hidden_dim * 4)  # 784 → 256

## fc1, 2, 3, classifier 의 역할(fc1~3이 각각의 레이어임)
self.fc1 = nn.Linear(28*28, hidden_dim * 4)
self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
self.classifier = nn.Linear(hidden_dim, num_classes)

############### PyTorch는 nn.Module을 상속받으면, 이런 self.fc1, self.fc2처럼 self.로 정의된 애들을 자동으로 인식해서 관리해줌 ############

nn.Linear(in_features, out_features)
## in_features : 입력데이터의 차원
## out_features : 출력 데이터의 차원
## fc1 ~ fc3 : 은닉층(입력을 계속 변형하면서 특징을 잘 뽑아냄)
## classifier : 최종 출력층(클래스 분류용)

## 1. fc1 : 28*28 = 784 픽셀짜리 이미지를 펼쳐서 넣어줌
## 2. fc1 : 784 차원 -> 큰 차원으로 확장( 더 많은 특징 만들기 )
## 3. fc2,3 : 점점 줄이면서 중요한 정보만 남김
## 4. classfier : 클래스 수 (예:  10개 숫자)만큼 결과 뽑음

## Batch Normalization
self.batchnorm1 = nn.BatchNorm1d(hidden_dim * 4)
## 각 층마다 값들의 분포를 평균 0, 분산1로 규정화 해주는 것
## 학습 중에 중간값이 너무 커지거나 작아지면 학습이 느려지고 불안정함
## BatchNorm은 그걸 막아줘서 속도를 빠르게 하고, 일정한 분포로 유지시켜줌

## 속도가 빨라지는 이유 : 
## 각 레이어 출력의 분포가 일정하게 유지됨
## 덕분에 큰 학습률로 학습해도 안정적임
## 깊은 네트워크도 빠르게 수렴함

nn.Identity()

## 아무것도 안하는 연산자
## if 문에서 옵션을 껐다 켰다 하려고 쓴는 것.

self.batchnorm1 = nn.BatchNorm1d(...) if apply_batchnorm else nn.Identity()

## True면 실제 BatchNorm 쓰고
## False면 그냥 값 그대로 통과시킴
## 이렇게 하면 코드를 간단하게 유지 가능

###### BatchNorm에서 “분포를 평균 0, 분산 1로 정규화”란?  -------> 쉽게 말하면, 출력되는 값들을 평균 0, 분산 1의 표준 정규분포처럼 바꿔준다는 뜻

self.dropout = nn.Dropout(p)
## 학습 중 무작위로 뉴런을 꺼버리는 것 (즉, 일부 출력을 0으로 만들기)
## 왜 그러냐?
## 모든 뉴런이 너무 협업하면 -> 과적합
## 일부러 랜덤하게 꺼주면 -> 다양한 조합을 학습하게 되어 일반화가 잘 됨
## 학습할떄만 작동하고, 테스트할 떄는 꺼짐.

self.activation = nn.ReLU()
## 0보다 작으면 0, 크면 그대로인 비선형 함수
ReLU(x) = max(0, x)
## 비선형 함수가 왜 필요할까
## 선형만 쓰면 아무리 층을 쌓아도 복잡한 구조 학습 못 하기 떄문
## ReLU 같은 비선형 함수가 들어가야 모델이 복잡한 문제를 학습 가능

####### 선형 함수: 직선처럼 비율대로 늘어나는 함수
####### 비선형 함수: 꺾이거나 굽어지는 함수

####### 딥러닝은 단순한 함수만 반복하면 아무리 레이어 쌓아도 똑같은 결과가 나와.
####### 그래서 중간에 꺾이는 “비선형 함수”를 써야 복잡한 데이터도 잘 표현할 수 있어!

####### ReLU는 계산이 빠르고, gradient vanishing(=미분이 작아져서 학습 멈춤) 문제도 줄어들어서 제일 많이 씀




## forward 함수 해부하기

def forward(self, x):
    x = x.view(x.shape[0], -1)
    x = self.batchnorm1(self.activation(self.fc1(x)))
    x = self.dropout(x)
    x = self.batchnorm2(self.activation(self.fc2(x)))
    x = self.dropout(x)
    x = self.batchnorm3(self.activation(self.fc3(x)))
    x = self.classifier(x)
    return x

## 1. Flatten (펴기)
x = x.view(x.shape[0], -1)
##  28x28 이미지를 784개 픽셀로 펼침
## x.shape[0]: 배치 크기 (예: 64)
## -1: 알아서 계산해서 784로 만들어줌

## 2. 은닉층 연산
x = self.fc1(x)
x = self.activation(x)
x = self.batchnorm1(x)
x = self.dropout(x)

## fc1: 첫 번째 완전연결 레이어
## activation: ReLU로 비선형성 부여
## batchnorm: 정규화
## dropout: 일부 뉴런 꺼줌

## 이런 구조가 fc2, fc3까지 반복됨

## 마지막 출력
x = self.classifier(x)
## 여기선 클래스 수만큼 결과를 내줌 (예: 10개 숫자 분류)

######################################
######## 반복형 구조 DNN 클래스 ########
######################################
class DNN(nn.Module):
  def __init__(self, hidden_dims, num_classes, dropout_ratio,
               apply_batchnorm, apply_dropout, apply_activation, set_super):
    if set_super:
      super().__init__()

    ## 784→512, 512→256, 256→128 순서로 Fully Connected Layer가 쌓임
    ## 즉, 은닉층을 3개 만들겠다는 뜻
    self.hidden_dims = hidden_dims

    ## 일반 Python list를 쓰면 안 되고 반드시 ModuleList여야 PyTorch가 내부 레이어를 추적함.
    self.layers = nn.ModuleList()

    ## 레이어 수가 많아도 hidden_dims만 바꾸면 구조 전체 자동 생성됨
    ## apply_batchnorm, apply_activation, apply_dropout을 조합해서 모듈처럼 생성
    ## Linear → BatchNorm → ReLU → Dropout 순으로 반복해서 쌓아
    for i in range(len(hidden_dims) - 1):
      self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

       # 선택적으로 배치 정규화 추가
      if apply_batchnorm:
        self.layers.append(nn.BatchNorm1d(hidden_dims[i+1]))

      # 선택적으로 활성화 함수 추가 (ReLU 사용)
      if apply_activation:
        self.layers.append(nn.ReLU())

      # 선택적으로 드롭아웃 추가 (과적합 방지용)
      if apply_dropout:
        self.layers.append(nn.Dropout(dropout_ratio))

    # 은닉층들을 모두 지난 후 마지막 출력층 정의
    # 마지막 hidden_dim → num_classes (예: 128 → 10)
    self.classifier = nn.Linear(hidden_dims[-1], num_classes)
    self.softmax = nn.LogSoftmax(dim=1)

  ## self.layers에 담긴 모든 레이어를 순서대로 거침
  ## 이 안에 Linear, BatchNorm, ReLU, Dropout 등이 이미 들어가 있음
  def forward(self, x):

   ## """
   ## Input and Output Summary

   ## Input:
   ##   x: [batch_size, 1, 28, 28]  ← MNIST처럼 28x28 흑백 이미지
   ## Output:
   ##   output: [batch_size, num_classes] ← 클래스별 확률 점수
   ## """

    # 2차원 이미지 텐서를 1차원 벡터로 펼침
    x = x.view(x.shape[0], -1)  # Flatten: [B, 1, 28, 28] → [B, 784]

    ## ModuleList에 들어있는 모든 레이어를 순서대로 거침
    ## 앞서 구성한 self.layers에 들어 있는 Linear, BatchNorm, ReLU, Dropout 레이어를 순서대로 통과
    for layer in self.layers:
      x = layer(x)

    # 마지막 출력층 통과
    x = self.classifier(x) # 예: [B, 128] → [B, 10]

    # log-softmax로 확률처럼 해석 가능한 출력값 생성
    output = self.softmax(x)

    return output

# --------------------------------------------
# 모델 생성 및 더미 입력 데이터로 동작 테스트
# --------------------------------------------

# 기본 hidden_dim 설정 (계층 너비 기준)
hidden_dim = 128

# hidden_dims는 입력층부터 은닉층 구조 정의
hidden_dims = [784, hidden_dim * 4, hidden_dim * 2, hidden_dim]
# → [784, 512, 256, 128]


# 앞서 만든 DNN 클래스 사용해서 모델 객체 생성
model = DNN(
    hidden_dims=hidden_dims,
    num_classes=10,           # MNIST: 0~9 숫자 10개 분류
    dropout_ratio=0.2,        # 드롭아웃 비율 20%
    apply_batchnorm=True,     # 배치정규화 적용
    apply_dropout=True,       # 드롭아웃 적용
    apply_activation=True,    # ReLU 활성화 함수 적용
    set_super=True            # nn.Module 초기화 실행
)


# 모델이 잘 작동하는지 확인: 더미 이미지 32개를 입력
output = model(torch.randn((32, 1, 28, 28)))  # [B=32, C=1, H=28, W=28]


# nn.Module의 초기화를 생략한 경우 (의도적 실수)
model = DNN(
    hidden_dims=hidden_dims,
    num_classes=10,
    dropout_ratio=0.2,
    apply_batchnorm=True,
    apply_dropout=True,
    apply_activation=True,
    set_super=False  # ❌ 이 경우 forward 등 내부 작동이 비정상일 수 있음
)
# 실제로 이 상태에서 model(...) 하면 AttributeError 혹은 작동 오류 발생



# --------------------------------------------
# weight_initialization() 함수 정의
# --------------------------------------------

# 모델의 모든 Linear 레이어 가중치를 초기화해주는 함수
def weight_initialization(model, weight_init_method):
    # model.modules()는 모델 안에 포함된 모든 레이어를 반환함 (중첩된 것도 포함)
    for m in model.modules():
        if isinstance(m, nn.Linear):  # Linear 레이어만 초기화 적용
            
            # 'gaussian' 방식: 평균 0, 표준편차 1의 정규분포로 가중치 초기화
            if weight_init_method == 'gaussian':
                nn.init.normal_(m.weight)

            # 'xavier' 방식: Xavier 초기화 (sigmoid, tanh에 적합)
            elif weight_init_method == 'xavier':
                nn.init.xavier_normal_(m.weight)

            # 'kaiming' 방식: Kaiming(He) 초기화 (ReLU에 적합)
            elif weight_init_method == 'kaiming':
                nn.init.kaiming_normal_(m.weight)

            # 'zeros' 방식: 가중치를 모두 0으로 설정 (보통은 잘 안 씀)
            elif weight_init_method == 'zeros':
                nn.init.zeros_(m.weight)

            # bias는 항상 0으로 초기화 (이건 보편적인 설정)
            nn.init.zeros_(m.bias)  # bias는 항상 0으로 초기화

    # 초기화가 완료된 model을 반환함
    return model

# --------------------------------------------
# 초기화 함수 호출 예시 + 초기화된 가중치 출력
# --------------------------------------------

# 초기화 방식 설정: 'gaussian', 'xavier', 'kaiming', 'zeros' 중 택 1
init_method = 'zeros'  # 적용할 초기화 방식 (예: xavier, kaiming 등)

# model 객체에 weight_initialization 함수를 적용
model = weight_initialization(model, init_method)

# model 내부에 있는 모든 모듈 중, Linear인 첫 번째 층의 가중치를 출력
for m in model.modules():
    if isinstance(m, nn.Linear):  # Linear 층이면
        print(m.weight.data)  # 초기화된 weight 값을 출력 (Tensor 형태)
        break # 첫 번째 것만 확인하면 되므로 break

## gaussian 방식 :
# 장점 
# 평균 0, 표준편차 1을 기본으로 하는 정규분포(가우시안 분포) 에서 값을 뽑아서 초기화
# 가장 단순한 방식이고, 초창기 신경망에서 많이 씀

# 단점 
# 층이 깊어질수록 값이 너무 커지거나 너무 작아짐 → 폭발/소실 문제

## xavier 방식 :
# 입력과 출력 뉴런 수를 고려해 가중치 분포를 자동으로 조절
# 분산을 1 / (입력노드수 + 출력노드수) 로 맞춰줌
# sigmoid, tanh 같은 대칭 비선형 함수에 적합
# 은닉층 깊어져도 폭발/소실이 덜함 → 안정적인 학습 가능

## kaiming 방식 : 
# ReLU 계열 함수(ReLU, LeakyReLU 등)에 최적화된 초기화 방법
# ReLU는 음수 입력을 0으로 만들기 때문에, 활성화 값이 사라지는 문제가 발생함
# 이를 보정하기 위해 입력 노드 수만 기준으로 분산을 조정

## zeros 방식 : 
# 모든 가중치를 0으로 설정
# 모든 뉴런이 같은 값을 학습하게 됨 (symmetry problem)
# 그래서 학습이 전혀 일어나지 않음
# 실습이나 실험용으로는 쓸 수 있지만, 실전에서는 절대 ❌