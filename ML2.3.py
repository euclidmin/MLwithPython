import mglearn
import matplotlib.pyplot as plt
import numpy as np


# #데이터 셋을 만듭니다.
# x, y = mglearn.datasets.make_forge()
# #산점도를 그립니다.
# mglearn.discrete_scatter(x[:, 0], x[:,1],y)
# plt.legend(["클래스 0", "클래스 1"], loc=4)
# plt.xlabel("첫 번째 특성")
# plt.ylabel("두 번째 특성")
# plt.show()
# print("x.shape:{}".format(x.shape))
#
#
# x, y = mglearn.datasets.make_wave(n_samples=40)
# plt.plot(x, y, 'o')
# plt.ylim(-3, 3)
# plt.xlabel("특성")
# plt.ylabel("타깃")
# plt.show()


#
# from sklearn.datasets import load_breast_cancer
# cancer = load_breast_cancer()
# print("cancer.keys(): \n{}".format(cancer.keys()))
# print("유방암 데이터의 형태:{}".format(cancer.data.shape))
# print("클래스별 샘플 개수:\{}".format(
#     {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}
# ))
# print(cancer.target_names)
# print(cancer.target)
# print("특성이름:\n{}".format(cancer.feature_names))




# from sklearn.datasets import load_boston
# boston = load_boston()
# print("데이터의 형태:{}".format(boston.data.shape))
#
# X, y = mglearn.datasets.load_extended_boston()
# print("X.shape:{}".format(X.shape))



#### 2.3.2 K-최근접 이웃
# mglearn.plots.plot_knn_classification(n_neighbors=1)
# mglearn.plots.plot_knn_classification(n_neighbors=3)


# from sklearn.model_selection import train_test_split
# X, y =  mglearn.datasets.make_forge()
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#
# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=3)
# clf.fit(X_train, y_train)
# print("테스트 세트 예측{}".format(clf.predict(X_test)))
# print("테스트 세트 정확도: {:.2f}".format(clf.score(X_test, y_test)))
#
#
# fig, axes = plt.subplots(1, 3, figsize=(10, 3))
#
# for n_neighbors, ax in zip([1, 3, 9], axes) :
#     clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
#     mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4 )
#     mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
#     ax.set_title("{} 이웃".format(n_neighbors))
#     ax.set_xlabel("특성 0")
#     ax.set_ylabel("특성 1")
# axes[0].legend(loc=3)
# plt.show()



# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
#
# cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=66)
#
# training_accuracy = []
# test_accuracy = []
#
# neighbors_settings = range(1, 11)
# for n_neightbors in neighbors_settings:
#     clf = KNeighborsClassifier(n_neighbors=n_neightbors)
#     clf.fit(X_train, y_train)
#     training_accuracy.append(clf.score(X_train, y_train))
#     test_accuracy.append(clf.score(X_test, y_test))
#
# plt.plot(neighbors_settings, training_accuracy, label="훈력 정확도")
# plt.plot(neighbors_settings, test_accuracy, label="테스트 정확도")
# plt.ylabel("정확도")
# plt.xlabel("n_neighbors")
# plt.legend()
# plt.show()

# mglearn.plots.plot_knn_regression(n_neighbors=1)
# mglearn.plots.plot_knn_regression(n_neighbors=3)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# reg = KNeighborsRegressor(n_neighbors=3)
# reg.fit(X_train, y_train)
# print("테스트 세트 예측:\n{}".format(reg.predict(X_test)))
# print("테스트 세트 R^2:{:.2f}".format(reg.score(X_test, y_test)))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes) :
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)

    ax.set_title(
        "{} 이웃의 훈련 스코어 : {:.2f} 테스크 스코어:{:.2f}".format(
            n_neighbors, reg.score(X_train, y_train),
            reg.score(X_test, y_test)
        )
    )
    ax.set_xlabel("특성")
    ax.set_ylabel("타깃")
axes[0].legend(["모델 예측", "훈련 데이터/타깃", "데스트 데이터/타깃"], loc="best")
plt.show()







