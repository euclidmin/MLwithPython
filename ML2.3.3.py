import mglearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# mglearn.plots.plot_linear_regression_wave()


# X, y = mglearn.datasets.make_wave(n_samples=60)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# lr = LinearRegression().fit(X_train, y_train)
# print("lr.coef_:{}".format(lr.coef_))
# print("lr.intercept_:{}".format(lr.intercept_))
# print("훈련 세트 점수: {:.2f}".format(lr.score(X_train, y_train)))
# print("테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))


X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)
print("lr.coef_:{}".format(lr.coef_))
print("lr.intercept_:{}".format(lr.intercept_))
print("훈련 세트 점수: {:.2f}".format(lr.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))


from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(ridge.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge.score(X_test, y_test)))

ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(ridge10.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge10.score(X_test, y_test)))

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(ridge01.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge01.score(X_test, y_test)))

import matplotlib.pyplot as plt

plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")

plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("계수 목록")
plt.ylabel("계수 크기")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()
plt.show()


from sklearn.linear_model import Lasso
import numpy as np

lasso = Lasso().fit(X_train, y_train)
print("훈련 세트 점수:{:.2f}".format(lasso.score(X_train, y_train)))
print("테스트 세트 점수{:.2f}".format(lasso.score(X_test, y_test)))
print("사용한 특성의 수{}".format(np.sum(lasso.coef_ != 0)))


lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("훈련 세트 점수:{:.2f}".format(lasso001.score(X_train, y_train)))
print("테스트 세트 점수{:.2f}".format(lasso001.score(X_test, y_test)))
print("사용한 특성의 수{}".format(np.sum(lasso001.coef_ != 0)))
print("사용한 특성의 수{}".format(np.sum(lasso001.coef_ != 0)))

lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("훈련 세트 점수:{:.2f}".format(lasso00001.score(X_train, y_train)))
print("테스트 세트 점수{:.2f}".format(lasso00001.score(X_test, y_test)))
print("사용한 특성의 수{}".format(np.sum(lasso00001.coef_ != 0)))
print("사용한 특성의 수{}".format(np.sum(lasso00001.coef_ != 0)))

plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.00001")
plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.xlabel("계수 목록")
plt.ylabel("계수 크기")
plt.show()



