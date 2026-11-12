import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fruits = np.load('fruits_300.npy')
#fruits 배열의 크기
print(fruits.shape)
#첫 번째 이미지의 첫 번째 행
print(fruits[0, 0, :])

"""
fig, axs = plt.subplots(1, 3)
axs[0].imshow(fruits[0], cmap='gray')
axs[1].imshow(fruits[100], cmap='gray')
axs[2].imshow(fruits[200], cmap='gray')
plt.show()
"""

# fruits 배열에서 순서대로 100개씩 선택
# reshape() 메서드를 사용해 두 번쨰 차원(100)과 세 번째 차원(100)을 10,000으로 합침
# 첫 번째 차원을 -1로 지정하면 자동으로 남은 차원 할당
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)
print(apple.shape)

# 사과 샘플 100개에 대한 픽셀 평균값
# 샘플의 값은 가로로 나열되어 있으므로 axis=1
print(apple.mean(axis=1))

#히스토그램 (값이 발생한 빈도를 표현한 그래프)
"""
plt.hist(apple.mean(axis=1), alpha=0.8, label='apple')
plt.hist(pineapple.mean(axis=1), alpha=0.8, label='pineapple')
plt.hist(banana.mean(axis=1), alpha=0.8, label='banana')
plt.legend()
plt.show()
"""

#샘플의 평균값이 아니라 픽셀별 평균값 비교
"""
fig, axs = plt.subplots(1, 3, figsize=(15, 3))
axs[0].bar(range(10000), apple.mean(axis=0))
axs[1].bar(range(10000), pineapple.mean(axis=0))
axs[2].bar(range(10000), banana.mean(axis=0))
plt.show()
"""

#픽셀을 평균 낸 이미지를 모든 사진을 합쳐 놓은 이미지
apple_mean = apple.mean(axis=0).reshape(100, 100)
pineapple_mean = pineapple.mean(axis=0).reshape(100, 100)
banana_mean = banana.mean(axis=0).reshape(100, 100)
"""
fig, axs = plt.subplots(1, 3, figsize=(15, 3))
axs[0].imshow(apple_mean, cmap='gray')
axs[1].imshow(pineapple_mean, cmap='gray')
axs[2].imshow(banana_mean, cmap='gray')
plt.show()
"""

# numpy.abs : 절댓값 계산 (np.absolute)
abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis = (1, 2))
print(abs_mean.shape)

# apple_mean과 오차가 가장 작은 샘플 100개
apple_index = np.argsort(abs_mean)[:100]
apple_index = apple_index.reshape(10, 10)
fig, axs = plt.subplots(10, 10, figsize = (10, 10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[apple_index[i, j]], cmap = 'gray')
        axs[i, j].axis('off')
plt.show()
