import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']]
target = wine['class']

train_input, test_input, train_target, test_target = train_test_split(data, target,test_size=0.2,random_state=42)

#검증 세트
sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)
print(sub_input.shape, val_input.shape)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input,sub_target)
print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))

scores = cross_validate(dt, train_input, train_target)
print(scores)

print(np.mean(scores['test_score']))

scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
print(np.mean(scores['test_score']))

splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))

params = {'min_impurity_decrease':[0.0001,0.0002,0.0003,0.0004,0.0005]}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
dt = gs.best_estimator_
print(dt.score(train_input, train_target))

#최적의 매개변수
print(gs.best_params_)

#각 매개변수에서 수행한 교차 검증의 평균 점수
print(gs.cv_results_['mean_test_score'])

#최상의 검증 점수를 만든 매개변수 조합
print(gs.cv_results_['params'][gs.best_index_])

#min_impurity_decrease : 노드를 분할하기 위한 불순도 최소량 지정 / max_depth : 트리의 깊이 제한 / min_sample_split : 노드를 나누기 위한 최소 샘플 수
params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001), 'max_depth':range(5, 20, 1), 'min_samples_split': range(2, 100, 10)}

"""
#verbose=1 몇 번째 작업 수행 중인지 표시
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1, verbose=1)
gs.fit(train_input, train_target)

#최상의 매개변수 조합
print(gs.best_params_)
#최상의 교차 검중 점수
print(np.max(gs.cv_results_['mean_test_score']))
"""
print("\n------------------------------------------------------------------------")
#균등 분포 샘플링(randint - 정수값)
rgen = randint(0, 10)
print(rgen.rvs(10))

print("\n------------------------------------------------------------------------")
#균등 분포 샘플링(randint - 정수값 / 1000개 샘플링)
print(np.unique(rgen.rvs(1000), return_counts=True))

print("\n------------------------------------------------------------------------")
#균등 분포 샘플링(uniform - 실숫값)
ugen = uniform(0, 1)
print(ugen.rvs(10))

params = {'min_impurity_decrease':uniform(0.0001,0.001),'max_depth':randint(20,50),'min_samples_split':randint(2,25),'min_samples_leaf':randint(1, 25)}
rs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=-1, random_state=42)
rs.fit(train_input,train_target)
print(rs.best_params_)
print(np.max(rs.cv_results_['mean_test_score']))
dt = rs.best_estimator_
print(dt.score(test_input, test_target))