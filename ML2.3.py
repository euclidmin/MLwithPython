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




from sklearn.datasets import load_boston
boston = load_boston()
print("데이터의 형태:{}".format(boston.data.shape))

X, y = mglearn.datasets.load_extended_boston()
print("X.shape:{}".format(X.shape))



#### 2.3.2 K-최근접 이웃
mglearn.plots.plot_knn_classification(n_neighbors=1)
