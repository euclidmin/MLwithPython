import mglearn

mglearn.plots.plot_scaling()

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)
print(X_train.shape)
print(X_test.shape)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

scalar = MinMaxScaler()
# scalar = StandardScaler()
# scalar = RobustScaler()
# scalar.fit(X_train)
print(scalar.fit(X_train))

# 데이터의 변환
X_train_scaled = scalar.transform(X_train)
X_test_scaled = scalar.transform(X_test)

# 데이터 변환 전과 후 비교
from sklearn.svm import SVC

# svm = SVC(C=100)
# svm.fit(X_train, y_train)
# print(svm.score(X_test, y_test))
#
# svm.fit(X_train_scaled, y_train)
# print(svm.score(X_test_scaled, y_test))
# # ==> 차이가 엄청나다.

svm = SVC(100)
svm.fit(X_train, y_train)
print(svm.score(X_test, y_test))

for scalar in [MinMaxScaler(), StandardScaler(), RobustScaler()]:
    scalar.fit(X_train)
    X_train_scaled = scalar.transform(X_train)
    X_test_scaled = scalar.transform(X_test)
    svm.fit(X_train_scaled, y_train)
    print(svm.score(X_test_scaled, y_test))
