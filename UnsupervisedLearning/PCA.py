import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from KMeans import draw_fruits
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

pca = PCA(n_components=50)
pca.fit(fruits_2d)
# PCA 클래스가 찾은 주성분
print(pca.components_.shape)
# 원본 데이터에서 가장 분산이 큰 방향을 순서대로 나타낸 것
#draw_fruits(pca.components_.reshape(-1, 100, 100))
# 원본 데이터 차원을 50으로 줄임
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)
# 차원 50으로 축소한 데이터를 복원
fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape)

"""
fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
for start in [0, 100, 200]:
    draw_fruits(fruits_reconstruct[start:start+100])
    print("\n")
"""
print(np.sum(pca.explained_variance_ratio_))

#plt.plot(pca.explained_variance_ratio_)
#plt.show()

lr = LogisticRegression()
# 지도학습을 위한 타깃 데이터 설정
target = np.array([0]*100 + [1]*100 + [2]*100)

scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores['test_score']))
# 각 교차 검증 폴드의 훈련 시간
print(np.mean(scores['fit_time']))

scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

# 설명된 분산의 50%에 달하는 주성분
pca = PCA(n_components=0.5)
pca.fit(fruits_2d)
print(pca.n_components_)
# 2개의 주성분 사용
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)

scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

# 차원 축소된 데이터를 사용해 k-평균 알고리즘 클러스터를 찾음
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)
print(np.unique(km.labels_, return_counts=True))
"""
for label in range(0, 3):
    draw_fruits(fruits[km.labels_ == label])
    print("\n")
"""

for label in range(0, 3):
    data = fruits_pca[km.labels_ ==label]
    plt.scatter(data[:,0], data[:,1])
plt.legend(['apple','banana','pineapple'])
plt.show()
