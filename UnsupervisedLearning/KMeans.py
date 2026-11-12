import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

fruits = np.load('fruits_300.npy')
# 3차원 배열 -> 2차원 배열
fruits_2d = fruits.reshape(-1, 100*100)

# n_clusters : 클러스터 개수
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)
# labels_ : 군집된 결과
print(km.labels_)

# 레이블 0, 1, 2로 모은 샘플의 개수
print(np.unique(km.labels_, return_counts=True))

def draw_fruits(arr, ratio = 1):
    n = len(arr)
    # 한 줄에 10개씩 이미지 그림
    # 샘플 개수를 10으로 나누어 전체 행 개수 계산
    rows = int(np.ceil(n/10))
    # 행이 1개면 열의 개수는 샘플 개수. 그렇지 않으면 10개
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, figsize = (cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n: # n개 까지만 그림
                axs[i, j].imshow(arr[i*10 + j], cmap = 'gray')
            axs[i, j].axis('off')
    plt.show()

#draw_fruits(fruits[km.labels_==0])
#draw_fruits(fruits[km.labels_==1])
#draw_fruits(fruits[km.labels_==2])

# cluster_centers_ : KMeans 클래스가 최종적으로 찾은 클러스터 중심
# 각 중심을 이미지로 출력하려면 100 x 100 2차원 배열로 변경해야 함
#draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)

# KMeans의 transform() : 훈련 데이터 샘플에서 클러스터 중심까지 거리로 변환
# 인덱스가 100인 샘플에 transform()
print(km.transform(fruits_2d[100:101]))
# 가장 가까운 클러스터 중심을 예측 클래스로 출력
# 레이블을 0으로 예측
print(km.predict(fruits_2d[100:101]))
#draw_fruits(fruits[100:101])
# 알고리즘 반복 횟수
print(km.n_iter_)

# 엘보우 방법
"""
inertia = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)
plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()
"""