import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
# 훈련 데이터의 크기
# 28x28크기인 60,000개의 이미지
#print(train_input.shape, train_target.shape)
# 테스트 세트의 크기
# 10,000개의 이미지
#print(test_input.shape, test_target.shape)
"""
fig, axs = plt.subplots(1, 10, figsize=(10, 10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray')
    axs[i].axis('off')
plt.show()
"""
#print(train_target[:10])

#print(np.unique(train_target, return_counts=True))
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
#print(train_scaled.shape)

sc = SGDClassifier(loss='log_loss', max_iter=5, random_state=42)
scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
#print(np.mean(scores['test_score']))



#print(keras.config.backend())

train_scaled, val_scaled, train_target, val_target, = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

#print(train_scaled.shape, train_target.shape)
#print(val_scaled.shape, val_target.shape)
# 이 예제의 샘플은 784개의 픽셀이 1차원으로 펼쳐저 있으므로 (784,)와 같이 배열의 크기를 지정
inputs = keras.layers.Input(shape=(784,))

# 밀집층(완결 연결층)
# 첫 번째 매개변수 : 뉴런 개수 10개 (패션 아이템 분류)
dense = keras.layers.Dense(10, activation='softmax')

# 입력층과 밀집층을 가진 신경망 모델
model = keras.Sequential([inputs, dense])

# 케라스 모델 훈련하기 전 설정 단계
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 이진 분류 : loss = 'binary_crossentropy'
# 다중 분류 : loss = 'categorical_crossentropy'

# 5번 반복에 정확도 약 85%
model.fit(train_scaled, train_target, epochs=5)
# 검증 세트에서 모델의 성능 확인 : evaluate()
model.evaluate(val_scaled, val_target)