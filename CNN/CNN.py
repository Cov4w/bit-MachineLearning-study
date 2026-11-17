import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

#keras.layers.Conv2D(10, kernel_size=(3,3), activation='relu')
# padding 매개변수로 패딩 지정 가능 (세임 패딩 : 'same')
#keras.layers.Conv2D(10, kernel_size=(3,3), activation='relu', padding='same')
# strides 매개변수 : 아래쪽으로 이동하는 크기와 오른쪽으로 이동하는 크기를 (1, 1)과 같이 튜플을 사용해 각각 지정할 수 있다.
#keras.layers.Conv2D(10, kernel_size=(3,3), activation='relu', padding='same', strides=1)

# 폴링 수행
# 첫 번째 매개변수로 폴링의 크기 지정 (대부분 폴링의 크기는 2)
# 예를 들어 (2, 3)로 지정하면 높이 방향으로 두칸, 너비 방향으로 세 칸씩 이동
#keras.layers.MaxPool2D(2)

#keras.layers.MaxPool2D(2, strides=2, padding='valid')

#---------------------------------------------------------------------------------
# 합성곱 신경망을 이용한 이미지 분류

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input.reshape(-1, 28, 28, 1)/255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

model = keras.Sequential()
model.add(keras.layers.Input(shape=(28,28,1)))
# 첫 번째 합성곱 층 Conv2D
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'))
# 풀링 층 전형적인 풀링 크기(2, 2)
# 패션 MNIST 이미지가 (28, 28) 크기에 세임 패딩을 적용했기 때문에 합성곱 층에서 출력된 특성 맵의 가로세로 크기는 동일
model.add(keras.layers.MaxPool2D(2))
# (2, 2) 풀링을 적용했으므로 특성 맵의 크기는 절반으로 줄어든다.
# 합성곱 층에서 32개의 필터를 사용했기 때문에 이 특성 맵의 깊이는 32가 된다.
# 따라서 최대 풀링은 통과한 특성 맵의 크기는 (14, 14, 32)가 될 것이다.

# 필터의 개수를 64개로 늘린 두 번째 합성곱-풀링 층
# 64개의 필터를 사용했으므로 최종적으로 만들어지는 특성 맵의 크기는 (7,7, 64)가 될 것이다.
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.MaxPool2D(2))

# 마지막 10개의 뉴런을 가진 (밀집) 출력층에서 확률을 계산하기 위해 3차원 특성 맵을 일렬로 펼치기
# Flatten -> Dense -> Dense
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation='softmax'))
# 드롭아웃 층이 은닉층의 과대적합을 막아 성능을 조금 더 개선해 줄 것입니다.
model.summary()

#print(keras.utils.plot_model(model))
#keras.utils.plot_model(model, show_shapes=True)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.keras', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
history = model.fit(train_scaled, train_target, epochs=20, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb, early_stopping_cb])

'''
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
'''
model.evaluate(val_scaled, val_target)

plt.imshow(val_scaled[0].reshape(28, 28), cmap='gray_r')
plt.show()

preds = model.predict(val_scaled[0:1])
print(preds)

plt.bar(range(1,11), preds[0])
plt.xlabel('class')
plt.ylabel('prob.')
plt.show()
# MNIST 데이터셋을 파이썬에서 레이블을 다루기 위해 리스트로 저장
classes = ['티셔츠', '바지', '스웨터', '드레스', '코트', '샌달', '셔츠', '스니커즈', '가방', '앵클 부츠']
# preds 배열에서 가장 큰 인덱스를 찾아 classes 리스트의 인덱스로 사용
print(classes[np.argmax(preds)])

# 픽셀값의 범위를 0~1 사이로 바꾸고 채널 자원을 추가하기 위해 이미지 크기를 (28, 28, 1)로 바꾼다.
test_scaled = test_input.reshape(-1, 28, 28, 1)/255.0
model.evaluate(test_scaled, test_target)