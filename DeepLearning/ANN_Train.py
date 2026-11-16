import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(28,28)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='relu'))
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

#model = model_fn()
# 드롭 아웃
model = model_fn(keras.layers.Dropout(0.3))

#model.summary()

#model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 옵티마이저 하이퍼파라미터(Adam)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# verbose : 훈련 과정 출력을 조절
#history= model.fit(train_scaled, train_target, epochs=5, verbose=0)
#epochs=20
#history= model.fit(train_scaled, train_target, epochs=20, verbose=0)
# 검증 데이터 전달

history= model.fit(train_scaled, train_target, epochs=11, verbose=0, validation_data=(val_scaled, val_target))
# .keras 확장자를 가진 파일에 필요한 정보를 모두 압축하여 저장
model.save('model-whole.keras')
# 훈련된 모델의 파라미터만 저장
model.save_weights('model.weights.h5')
#print(history.history.keys())

# history 객체에는 훈련 측정값이 담겨 있는 history 딕셔너리가 들어있다.
"""
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

plt.plot(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
"""
model = model_fn(keras.layers.Dropout(0.3))
model.load_weights('model.weights.h5')
val_labels = np.argmax(model.predict(val_scaled), axis=-1)
print(np.mean(val_labels == val_target))

model = keras.models.load_model('model-whole.keras')
print(model.evaluate(val_scaled, val_target))

# 조기종료
# 과대적합이 시작되기 전에 훈련을 미리 중지
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.keras', save_best_only=True)
model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb])
model = keras.models.load_model('best-model.keras')
print(model.evaluate(val_scaled, val_target))

# EarlyStopping 콜백 (patience 매개변수)
# 검증 점수가 향상되지 않더라도 참을 에포크 횟수로 지정
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best=model.keras', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb, early_stopping_cb])
print(early_stopping_cb.stopped_epoch)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label = 'val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

model.evaluate(val_scaled, val_target)