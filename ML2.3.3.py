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


# X, y = mglearn.datasets.load_extended_boston()
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# lr = LinearRegression().fit(X_train, y_train)
# print("lr.coef_:{}".format(lr.coef_))
# print("lr.intercept_:{}".format(lr.intercept_))
# print("훈련 세트 점수: {:.2f}".format(lr.score(X_train, y_train)))
# print("테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))
#
#
# from sklearn.linear_model import Ridge
#
# ridge = Ridge().fit(X_train, y_train)
# print("훈련 세트 점수: {:.2f}".format(ridge.score(X_train, y_train)))
# print("테스트 세트 점수: {:.2f}".format(ridge.score(X_test, y_test)))
#
# ridge10 = Ridge(alpha=10).fit(X_train, y_train)
# print("훈련 세트 점수: {:.2f}".format(ridge10.score(X_train, y_train)))
# print("테스트 세트 점수: {:.2f}".format(ridge10.score(X_test, y_test)))
#
# ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
# print("훈련 세트 점수: {:.2f}".format(ridge01.score(X_train, y_train)))
# print("테스트 세트 점수: {:.2f}".format(ridge01.score(X_test, y_test)))
#
# import matplotlib.pyplot as plt
#
# plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
# plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
# plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
#
# plt.plot(lr.coef_, 'o', label="LinearRegression")
# plt.xlabel("계수 목록")
# plt.ylabel("계수 크기")
# plt.hlines(0, 0, len(lr.coef_))
# plt.ylim(-25, 25)
# plt.legend()
# plt.show()
#
#
# from sklearn.linear_model import Lasso
# import numpy as np
#
# lasso = Lasso().fit(X_train, y_train)
# print("훈련 세트 점수:{:.2f}".format(lasso.score(X_train, y_train)))
# print("테스트 세트 점수{:.2f}".format(lasso.score(X_test, y_test)))
# print("사용한 특성의 수{}".format(np.sum(lasso.coef_ != 0)))
#
#
# lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
# print("훈련 세트 점수:{:.2f}".format(lasso001.score(X_train, y_train)))
# print("테스트 세트 점수{:.2f}".format(lasso001.score(X_test, y_test)))
# print("사용한 특성의 수{}".format(np.sum(lasso001.coef_ != 0)))
# print("사용한 특성의 수{}".format(np.sum(lasso001.coef_ != 0)))
#
# lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
# print("훈련 세트 점수:{:.2f}".format(lasso00001.score(X_train, y_train)))
# print("테스트 세트 점수{:.2f}".format(lasso00001.score(X_test, y_test)))
# print("사용한 특성의 수{}".format(np.sum(lasso00001.coef_ != 0)))
# print("사용한 특성의 수{}".format(np.sum(lasso00001.coef_ != 0)))
#
# plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
# plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
# plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.00001")
# plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
# plt.xlabel("계수 목록")
# plt.ylabel("계수 크기")
# plt.show()

# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
# import matplotlib.pyplot as plt
#
# X, y = mglearn.datasets.make_forge()
# fig, axes = plt.subplots(1, 2, figsize=(10, 3))
#
# for model, ax in zip([LinearSVC(), LogisticRegression()], axes) :
#     clf = model.fit(X, y)
#     mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=.7)
#     mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
#     ax.set_title("{}".format(clf.__class__.__name__))
#     ax.set_xlabel("특성 0")
#     ax.set_ylabel("특성 1")
# axes[0].legend()
# plt.show()



# from sklearn.datasets import load_breast_cancer
# from sklearn.linear_model import LogisticRegression
# import matplotlib.pyplot as plt
#
# cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
#
# # logreg = LogisticRegression().fit(X_train, y_train)
# # print("훈련 셋트 점수:{:.3f}".format(logreg.score(X_train, y_train)))
# # print("테스트 세트 점수:{:.3f}".format(logreg.score(X_test, y_test)))
# #
# # logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
# # print("훈련 셋트 점수:{:.3f}".format(logreg100.score(X_train, y_train)))
# # print("테스트 세트 점수:{:.3f}".format(logreg100.score(X_test, y_test)))
# #
# # logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
# # print("훈련 셋트 점수:{:.3f}".format(logreg001.score(X_train, y_train)))
# # print("테스트 세트 점수:{:.3f}".format(logreg001.score(X_test, y_test)))
#
# C_ = [0.001, 1, 100]
# marker_ = ['o', '^', 'v']
#
# for C, marker in zip(C_, marker_):
#     logreg = LogisticRegression(C=C, penalty="l1").fit(X_train, y_train)
#     print("훈련 셋트 점수:{:.3f}".format(logreg.score(X_train, y_train)))
#     print("테스트 세트 점수:{:.3f}".format(logreg.score(X_test, y_test)))
#     plt.plot(logreg.coef_.T, marker, label="C={:.3f}".format(logreg.C))
# plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
# plt.hlines(0, 0, cancer.data.shape[1])
# plt.ylim(-5, 5)
# plt.xlabel("특성")
# plt.ylabel("계수 크기")
# plt.legend()
# plt.show()

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:, 0], X[:,1], y)
plt.xlabel("특성 0")
plt.ylabel("특성 1")
plt.legend(["클래스 0", "클래스 1", "클래스 2"])
plt.show()

from sklearn.svm import LinearSVC
import numpy as np
linear_svm = LinearSVC().fit(X, y)
print("계수 배열의 크기: ", linear_svm.coef_.shape)
print("절편 배열의 크기: ", linear_svm.intercept_.shape)

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, mglearn.cm3.colors):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)

plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel("특성 0")
plt.ylabel("특성 1")
plt.legend(['클래스 0', '클래스 1', '클래스 2', '클래스 0 경계', '클래스 1 경계', '클래스 2 경계'])
plt.show()

    

