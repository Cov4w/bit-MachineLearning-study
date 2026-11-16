from torchvision.datasets import FashionMNIST
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torchinfo import summary
import torch
import torch.optim as optim

fm_train = FashionMNIST(root='.', train=True, download=True)
fm_test = FashionMNIST(root='.', train=False, download=True)

#print(type(fm_train.data))
#print(fm_train.data.shape, fm_test.data.shape)
#print(fm_train.targets.shape, fm_test.targets.shape)

train_input = fm_train.data
train_target = fm_train.targets
train_scaled = train_input / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

#print(train_scaled.shape, val_scaled.shape)
# 파이토치에서는 모델의 입력 크기를 사전에 지정할 필요가 없습니다. 따라서 케라스의 Input()과 같은 별도의 입력 정의 함수가 없습니다.
# 케라스의 Dense 층과 동일한 역할을 하는 것이 파이토치의 Linear 층입니다. Linear 층을 사용할 때는 입력 크기와 출력 크기(뉴런 개수)를 매개변수로 전달해야 합니다.
# 파이토치에서는 활성화 함수를 별도의 층으로 추가해야 합니다. 렐루 함수의 경우 ReLu 층을 사용합니다.
# 출력츠에 해당하는 두 번째 Linear 층 다음에는 활성화 함수가 없습니다. 케라스에서는 다중 분류 문제를 해결하기 위해 마지막 층에 소프트맥스 함수를 포함했지만, 파이토치에서는 이를 생략합니다.

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

#summary(model, input_size=(32, 28, 28))

# GPU 사용 여부 설정
device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

epochs = 5
batches = int(len(train_scaled)/32)
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for i in range(batches):
        inputs = train_scaled[i*32:(i+1)*32].to(device)
        targets = train_target[i*32:(i+1)*32].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"에포크:{epoch+1}, 손실 : {train_loss/batches:.4f}")

model.eval()
with torch.no_grad():
    val_scaled = val_scaled.to(device)
    val_target = val_target.to(device)
    outputs = model(val_scaled)
    predicts = torch.argmax(outputs, 1)
    corrects = (predicts == val_target).sum().item()

accuracy = corrects / len(val_target)
print(f"검증 정확도 : {accuracy:.4f}")