import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST

model = nn.Sequential()
model.add_module('conv1', nn.Conv2d(1, 32, kernel_size=3, padding='same'))
model.add_module('relu1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(2))
model.add_module('conv2', nn.Conv2d(32, 64, kernel_size=3, padding='same'))
model.add_module('relu2', nn.ReLU())
model.add_module('pool2', nn.MaxPool2d(2))
model.add_module('flatten', nn.Flatten())
model.add_module('dense1', nn.Linear(3136, 100))
model.add_module('relu3', nn.ReLU())
model.add_module('dropout', nn.Dropout(0.3))
model.add_module('dense2', nn.Linear(100, 10))

# best_cnn_model.pt에 저장한 가중치 로드
model.load_state_dict(torch.load('best_cnn_model.pt', weights_only=True))
# model 객체의 children() 메서드는 모델에 추가된 층을 반환하는 파이썬 제너레이터 객체를 반환
layers = [layer for layer in model.children()]
# layers 리스트를 사용해 model의 층을 참조할 수 있다.
print(layers[0])
# Sequentail 클래스로 만든 모델은 정수 인덱스로 하위 층을 참조할 수 있다.
print(model[0])
# 모델 객체의 named_children() 메서드를 사용하면 층의 이름과 층 객체를 함께 얻을 수 있다.
for name, layer in model.named_children():
    print(f"{name:10s}", layer)
# 이름이 'conv1'인 첫 번째 합성곱 층은 다음과 같이 참조할 수 있다.
print(model.conv1)

# 파이토치 층의 가중치와 절편은 각각 weight와 bias 속성에 저장되어 있다.
# 이 속성은 파이토치의 Parameter 클래스의 객체이다.
conv_weights = model.conv1.weight.data
print(conv_weights.mean(), conv_weights.std())

plt.hist(conv_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

print(conv_weights.shape)

fig, axs = plt.subplots(2, 16, figsize = (15,2))
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(conv_weights[i*16+j,0,:,:], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')
plt.show()

fm_train = FashionMNIST(root='.', train=True, download=True)
train_input=fm_train.data
plt.imshow(train_input[0], cmap='gray_r')
plt.show()

ankle_boot = train_input[0:1].reshape(1, 1, 28, 28)/255.0
model.eval()
with torch.no_grad():
    feature_maps = model.conv1(ankle_boot)
    feature_maps = model.relu1(feature_maps)
print(feature_maps.shape)

fig, axs = plt.subplots(4, 8, figsize=(15,8))
for i in range(4):
    for j in range(8):
        axs[i,j].imshow(feature_maps[0,i*8+j,:,:])
        axs[i,j].axis('off')
plt.show()

model.eval()
with torch.no_grad():
    feature_maps = model.conv1(ankle_boot)
    feature_maps = model.relu1(feature_maps)
    feature_maps = model.pool1(feature_maps)
    feature_maps = model.conv2(feature_maps)
    feature_maps = model.relu2(feature_maps)

model.eval()
x=ankle_boot
with torch.no_grad():
    for name, layer in model.named_children():
        x=layer(x)
        if name == 'relu2':
            break
feature_maps = x

fig, axs = plt.subplots(8, 8, figsize=(12,12))
for i in range(8):
    for j in range(8):
        axs[i,j].imshow(feature_maps[0,i*8+j,:,:])
        axs[i,j].axis('off')
plt.show()