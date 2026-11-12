import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

wine = pd.read_csv('https://bit.ly/wine_csv_data')
print(wine.head())

#데이터프레임의 각 열의 데이터 타입과 누락된 데이터가 있는지 확인
wine.info()

#열에 대한 간략한 통계 출력
print(wine.describe())

data = wine[['alcohol','sugar','pH']]
target = wine['class']
#훈련 세트 / 테스트 세트 나누기 (20% 테스트 세트)
train_input, test_input, train_target, test_target = train_test_split(data, target,test_size=0.2,random_state=42)
print(train_input.shape, test_input.shape)

#StandardScaler를 이용해 훈련 세트 전처리
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

#로지스틱 회구 모델 훈련
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled,train_target))
print(lr.score(test_scaled,test_target))

#학습한 계수 / 절편
print(lr.coef_, lr.intercept_)

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target)) # 훈련 세트
print(dt.score(test_scaled, test_target))   # 테스트 세트

plt.figure(figsize=(15,10))
plot_tree(dt, filled=True,feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target)) # 훈련 세트
print(dt.score(test_input, test_target))   # 테스트 세트
plt.figure(figsize=(15, 10))

plot_tree(dt, filled=True,feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

print(dt.feature_importances_)