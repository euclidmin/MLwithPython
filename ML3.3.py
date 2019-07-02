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

# svm = SVC(100)
# svm.fit(X_train, y_train)
# print(svm.score(X_test, y_test))
#
# for scalar in [MinMaxScaler(), StandardScaler(), RobustScaler()]:
#     scalar.fit(X_train)
#     X_train_scaled = scalar.transform(X_train)
#     X_test_scaled = scalar.transform(X_test)
#     svm.fit(X_train_scaled, y_train)
#     print(svm.score(X_test_scaled, y_test))


from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# cancer =  load_breast_cancer()
#
# scaler = StandardScaler()
# scaler.fit(cancer.data)
# X_scaled = scaler.transform(cancer.data)
#
# pca = PCA(n_components=2)
# pca.fit(X_scaled)
#
# X_pca = pca.transform(X_scaled)
#
# print(X_scaled.shape)
# print(X_pca.shape)


# LFW 데이터 셋에 있는 이미지 샘플
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])
plt.show()

import numpy as np

counts = np.bincount(people.target)
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count))
