from keras.datasets import imdb
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
import keras

# keras.datasets 패키지 아래 imdb 모듈 임포트 후 자주 등장하는 200개 사용
(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=200)
print(train_input.shape, test_input.shape)
# 첫 번째, 두 번째 리뷰 길이 출력
print(len(train_input[0]))
print(len(train_input[1]))
# 첫 번째 리뷰에 담긴 내용 출력
print(train_input[0])
# 타깃값은 0 (부정)과 1 (긍정)로 나누어집니다.
print(train_target[:20])

train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)
# mean() 함수와 median() 함수를 사용해 리뷰 길이의 평균과 중간값 구하기
lengths = np.array([len(x) for x in train_input])
print(np.mean(lengths), np.median(lengths))

plt.hist(lengths)
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()

train_seq = pad_sequences(train_input, maxlen=100)
print(train_seq.shape)

# train_seq에 있는 첫 번째 샘플 출력
print(train_seq)
# train_input에 있는 원본 샘플의 끝 확인
print(train_input[0][-10:])
# 검증 세트의 길이 100으로 설정
val_seq = pad_sequences(val_input, maxlen=100)
'''
model = keras.Sequential()
# 샘플의 길이를 100으로 지정했기 때문에 첫 번째 매개변수 값 100
# imdb.load_data() 함수에서 200개의 단어만 사용하도록 지정했기 때문에 고유한 단어는 모두 200개
# 훈련 데이터에 포함될 수 있는 정숫값의 범위는 0(패딩 토큰)에서 199까지 때문에 두 번째 매개변수 값 200
model.add(keras.layers.Input(shape=(100,200)))
model.add(keras.layers.SimpleRNN(8))
model.add(keras.layers.Dense(1,activation='sigmoid'))

# train_seq를 원-핫 인코딩으로 변환하여 train_oh 배열로 만듦
train_oh = keras.utils.to_categorical(train_seq)
print(train_oh.shape)
# train_oh의 첫 번째 샘플의 첫 번째 토큰 10이 잘 인코딩되었는지 출력
print(train_oh[0][0][:12])
# 모든 원소의 값을 더해서 1이 되는지 확인
print(np.sum(train_oh[0][0]))
# val_seq도 원-핫 인코딩
val_oh = keras.utils.to_categorical(val_seq)

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.keras', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model.fit(train_oh, train_target, epochs=100, batch_size=64, validation_data=(val_oh, val_target), callbacks=[checkpoint_cb, early_stopping_cb])

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

print(train_seq.nbytes, train_oh.nbytes)
'''
(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)
train_seq=pad_sequences(train_input, maxlen=100)
val_seq = pad_sequences(val_input, maxlen=100)

# Embedding 클래스를 SimpleRNN 층 앞에 추가한 두 번째 순환 신경망
model_emb = keras.Sequential()
# train_seq를 그대로 사용하므로 Input 함수에 지정할 입력의 크기는 (100,)
model_emb.add(keras.layers.Input(shape=(100,)))
# Embedding 클래스의 첫 번째 매개변수(500)는 어휘 사전의 크기
# 두 번째 매개면수(16)는 임베딩 벡터의 크기이다.
model_emb.add(keras.layers.Embedding(500, 16))
model_emb.add(keras.layers.SimpleRNN(8))
model_emb.add(keras.layers.Dense(1, activation='sigmoid'))

model_emb.summary()

model_emb.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-embedding-model.keras', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)
history = model_emb.fit(train_seq, train_target, epochs=100, batch_size=64, validation_data=(val_seq, val_target), callbacks=[checkpoint_cb, early_stopping_cb])

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()