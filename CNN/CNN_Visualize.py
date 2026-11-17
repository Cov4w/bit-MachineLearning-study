import keras
import matplotlib.pyplot as plt
# 체크포인트 파일 읽어오기
model = keras.models.load_model('best-cnn-model.keras')

# model.layers 리스트에 이전 절에서 추가했던 Conv2D, MaxPooling2D 층이 번갈아 2번 연속 등장합니다.
print(model.layers)

# layers 속성의 첫 번째 원소를 선택해 weights의 첫 번째 원소(가중치)와 두 번째 원소(절편)의 크기 출력
conv = model.layers[0]
print(conv.weights[0].shape, conv.weights[1].shape)
# 이전 절에서 커널 크기를 (3, 3)으로 지정하고 이 합성곱 층에 전달되는 입력의 깊이가 1이므로 실제 커널의 크기는 (3, 3, 1)입니다.
# 또한 필터의 개수가 32개이므로 weights의 첫 번째 원소인 가중치의 크기는 (3, 3, 1, 32)기 되었습니다.
# weights의 두 번째 원소는 절편의 개수를 나타내는데 필터마다 1개의 절편이 있으므로 (32, ) 크기가 됩니다.

# weights 속성은 텐서플로의 다차원 배열인 Tensor 클래스의 객체입니다.
# numpy() 메서드를 사용해 넘파이 배열로 변환한 뒤 가중치 배열의 평균과 표준편차를 넘파이 mean() 메서드와 std() 메서드로 계산한다.
conv_weights = conv.weights[0].numpy()
print(conv_weights.mean(), conv_weights.std())
'''
plt.hist(conv_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()


fig, axs = plt.subplots(2, 16, figsize=(15,2))
for i in range(2):
    for j in range(16):
        # 
        axs[i, j].imshow(conv_weights[:,:,0,i*16+j], vmin = -0.5, vmax=0.5)
        axs[i,j].axis('off')
plt.show()
'''
no_training_model = keras.Sequential()
no_training_model.add(keras.layers.Input(shape=(23, 23, 1)))
no_training_model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'))
no_training_conv = no_training_model.layers[0]
print(no_training_conv.weights[0].shape)

no_training_weights = no_training_conv.weights[0].numpy()
print(no_training_weights.mean(), no_training_weights.std())
'''
plt.hist(no_training_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

fig, axs = plt.subplots(2, 16, figsize=(15,2))
for i in range(2):
    for j in range(16):
        # 
        axs[i, j].imshow(no_training_weights[:,:,0,i*16+j], vmin = -0.5, vmax=0.5)
        axs[i,j].axis('off')
plt.show()
'''
inputs = keras.Input(shape=(784,))
dense1 = keras.layers.Dense(100, activation='relu')
dense2 = keras.layers.Dense(10, activation='softmax')
# 함수처럼 호출할 수 있다.
hidden = dense1(inputs)
# 두 번째 층 호출, 첫 번째 층의 출력을 입력으로
outputs = dense2(hidden)
# inputs와 outputs을 Model 클래스로 연결
func_model = keras.Model(inputs, outputs)
print(model.inputs)

conv_acti = keras.Model(model.inputs[0], model.layers[0].output)

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
plt.imshow(train_input[0], cmap ='gray_r')
plt.show()

ankle_boot = train_input[0:1].reshape(-1, 28, 28, 1)/255.0
feature_maps = conv_acti.predict(ankle_boot)
print(feature_maps.shape)

fig, axs = plt.subplots(4, 8, figsize = (15, 8))
for i in range(4):
    for j in range(8):
        axs[i,j].imshow(feature_maps[0,:,:,i*8+j])
        axs[i, j].axis('off')
plt.show()

# model 객체의 입력과 두 번째 합성곱 층인 model.layers[2]의 출력을 연결한 conv2_acti 모델
conv2_acti = keras.Model(model.inputs[0], model.layers[2].output)
# 앵클 부츠 샘플을 conv2_acti 모델의 predict() 메서드에 전달
feature_maps = conv2_acti.predict(ankle_boot)
# 첫 번째 풀링 층에서 가로세로 크기가 절반으로 줄었고, 두 번째 합성곱 층의 필터 개수는 64개이므로 feature_maps의 크기는 배치 차원을 제외하면 (14, 14, 64)일 것이다.
print(feature_maps.shape)
# 64개의 특성 맵을 8개씩 나누어 imshow() 함수로 그리기
fig, axs = plt.subplots(8, 8, figsize=(12,12))
for i in range(8):
    for j in range(8):
        axs[i,j].imshow(feature_maps[0,:,:,i*8+j])
        axs[i,j].axis('off')
plt.show()