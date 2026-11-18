import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from keras.datasets import imdb
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader

(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target,test_size=0.2,random_state=42)

train_seq = pad_sequences(train_input, maxlen=100)
val_seq = pad_sequences(val_input, maxlen=100)

print(train_seq.shape, train_target.shape)

# 파이토치의 데이터로데에 사용하려면 케라스로 준비한 넘파이 배열 데이터를 파이토치 텐서로 변환해야합니다.
train_seq = torch.tensor(train_seq)
val_seq = torch.tensor(val_seq)

print(train_target.dtype)

train_target = torch.tensor(train_target, dtype=torch.float32)
val_target = torch.tensor(val_target, dtype=torch.float32)

print(train_target.dtype)

train_dataset = TensorDataset(train_seq, train_target)
val_dataset = TensorDataset(val_seq, val_target)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

class IMDBRnn(nn.Module):
    def __init__(self):
        super().__init__()
        # 첫 번째 매개변수 : 어휘 사전의 크기
        # 두 번째 매개변수 : 임베딩 벡터의 크기
        self.embedding = nn.Embedding(500, 16)
        # 첫 번째 매개변수 : 입력 크기(임베딩 벡터의 크기)
        # 두 번째 매개변수 : 출력 크기(뉴런 개수)
        # 배치 차원이 맨 앞이라는 것을 알리기 위해 batch_first 매개변수를 True로 지정
        self.rnn = nn.RNN(16, 8, batch_first=True)
        # 입력 크기가 8이고 출력 크기가 1인 nn.Linear 층
        self.dense = nn.Linear(8, 1)
        # 이진 분류이므로 마지막에 softmax 함수를 놓지 않았음
        self.sigmoid = nn.Sigmoid()

    # __init__() 메서드에서 정의한 층을 사용해 입력에서 출력까지 층 객체를 호출한다.
    def forward(self, x):
        x = self.embedding(x)
        # 각 타임스텝의 은닉 상태는 불필요하므로 self.rnn 객체가 반환되는 두 번째 값만 hidden 변수에 저장해서 사용
        _, hidden = self.rnn(x)
        outputs = self.dense(hidden[-1])
        return self.sigmoid(outputs)

model = IMDBRnn()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 손실 함수와 옵티마이저 정의
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 2e-4)

train_hist = []
val_hist = []
patience = 2
best_loss = -1
early_stopping_counter = 0

epochs = 100
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item()

    train_loss = train_loss/len(train_loader)
    val_loss = val_loss/len(val_loader)
    train_hist.append(train_loss)
    val_hist.append(val_loss)
    print(f"에포크:{epoch+1},",
          f"훈련 손실:{train_loss:.4f}, 검증 손실:{val_loss:.4f}")

    if best_loss == -1 or val_loss < best_loss:
        best_loss = val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), 'best_rnn_model.pt')
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print(f"{epoch+1}번째 에포크에서 조기 종료되었습니다.")
            break

plt.plot(train_hist, label='train')
plt.plot(val_hist, label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

model.load_state_dict(torch.load('best_rnn_model.pt', weights_only=True))

model.eval()
corrects = 0
with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        predicts = outputs > 0.5
        corrects += (predicts.squeeze() == targets).sum().item()

accuracy = corrects / len(val_dataset)
print(f"검증 정확도: {accuracy:.4f}")