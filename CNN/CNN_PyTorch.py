from torchvision.datasets import FashionMNIST
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

fm_train = FashionMNIST(root='.', train=True, download=True)
fm_test = FashionMNIST(root='.', train=False, download=True)

train_input = fm_train.data
train_target = fm_train.targets
train_scaled = train_input.reshape(-1, 1, 28, 28)/255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

model = nn.Sequential()
# 32개의 필터를 가진 합성곱 층과 렐루 활성화 함수, 풀링 층 추가
model.add_module('conv1', nn.Conv2d(1, 32, kernel_size=3, padding='same'))
model.add_module('relu1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(2))
# 이전 절에서 사용했던 렐루 활성화 함수 추가, 풀링 층 추가
# 마지막에 Faltten 층 추가
# 두 번째 합성곱 층의 입력 채널 크기는 첫 번째 합성곱 층의 출력 채널 개수와 같다.
model.add_module('conv2', nn.Conv2d(32, 64, kernel_size=3, padding='same'))
model.add_module('relu2', nn.ReLU())
model.add_module('pool2', nn.MaxPool2d(2))
model.add_module('flatten', nn.Flatten())
# 두 번째 합성곱 층의 출력을 1차원으로 펼쳤으므로, 이제 Linear 층을 추가한다.
# Linear 층을 사용할 때는 입력 크기를 첫 번째 매개변수로 전달해야 하므로, Flatten 층으로 펼친 입력의 크기를 먼저 알아야 한다.
#outputs = model(torch.ones(1, 1, 28, 28))
#print(outputs.shape)
# 출력 : torch.Size([1, 3136])
model.add_module('dense1', nn.Linear(3136, 100))
model.add_module('relu3', nn.ReLU())
model.add_module('dropout', nn.Dropout(0.3))
model.add_module('dense2', nn.Linear(100, 10))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# TensorDataset 클래스는 여러 텐서를 결합하여 하나의 데이터셋으로 관리하도록 도와준다.
train_dataset = TensorDataset(train_scaled, train_target)
val_dataset = TensorDataset(val_scaled, val_target)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

train_hist = []
val_hist = []
patience = 2
best_loss = -1
early_Stopping_counter = 0

epochs = 20
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    train_loss = train_loss/len(train_loader)
    val_loss = val_loss/len(val_loader)
    train_hist.append(train_loss)
    val_hist.append(val_loss)
    print(f"에포크:{epoch+1},", f"훈련 손실:{train_loss:.4f}, 검증 손실:{val_loss:.4f}")
    if best_loss == -1 or val_loss < best_loss:
        best_loss = val_loss
        early_Stopping_counter = 0
        torch.save(model.state_dict(), 'best_cnn_model.pt')
    else:
        early_Stopping_counter +=1
        if early_Stopping_counter >=patience:
            print(f"{epoch+1}번째 에포크에서 조기 종료되었습니다.")
            break
'''
plt.plot(train_hist, label='train')
plt.plot(val_hist, label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
'''
model.load_state_dict(torch.load('best_cnn_model.pt', weights_only=True))

model.eval()
corrects = 0
with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        predicts = torch.argmax(outputs, 1)
        corrects += (predicts == targets).sum().item()
accuracy = corrects/len(val_dataset)
print(f"검증 정확도 : {accuracy:.4f}")

test_scaled=fm_test.data.reshape(-1, 1, 28, 28)/255.0
test_target=fm_test.targets

test_dataset = TensorDataset(test_scaled, test_target)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
corrects = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        predicts = torch.argmax(outputs, 1)
        corrects += (predicts == targets).sum().item()
accuracy = corrects/len(test_dataset)
print(f"테스트 정확도 : {accuracy:.4f}")