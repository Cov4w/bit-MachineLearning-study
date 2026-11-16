import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
"""
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)
"""
# 시그모이드 활성화 함수를 사용한 은닉층과 소프트맥스 함수를 사용한 출력층을 케라스의 Dense 클래스로 만들기
#inputs = keras.layers.Input(shape=(784,))
# 은닉층 100개, 활성화 함수 : sigmoid
# 은닉층의 뉴런 개수는 적어도 출력층의 뉴런보다는 많게 만들어야 함.
#dense1 = keras.layers.Dense(100, activation='sigmoid')
# 10개의 쿨래스를 분류하므로 10개의 뉴런, 활성화 함수 : 소프트맥스 함수
#dense2 = keras.layers.Dense(10, activation='softmax')

# 심층 신경망
# Sequential 클래스의 객체를 만들 때 여러 개의 층을 추가하려면 리스트에 계속 플요한 층을 추가하면 됨.
# 입력층을 앞에 두고 출력층을 가장 마지막에 두어야 함
#model = keras.Sequential([inputs, dense1, dense2])

#model.summary()

#model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#model.fit(train_scaled, train_target, epochs=5)

# Flatten 클래스
# 배치 차원을 제외하고 나머지 입력 차원을 모두 일렬로 펼치는 역할
"""
model = keras.Sequential()
model.add(keras.layers.Input(shape=(28,28)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()
"""
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)
"""
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_scaled, train_target, epochs=5)

model.evaluate(val_scaled, val_target)
"""

model = keras.Sequential()
model.add(keras.layers.Input(shape=(28,28)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_scaled, train_target, epochs=5)

model.evaluate(val_scaled, val_target)