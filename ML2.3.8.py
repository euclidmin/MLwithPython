from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons

# X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
#
# mlp = MLPClassifier(solver='lbfgs', random_state=0)
# mlp.fit(X_train, y_train)
#
# print(mlp.score(X_train, y_train))
# print(mlp.score(X_test, y_test))

# X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
#
# mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10])
# mlp.fit(X_train, y_train)
#
# print(mlp.score(X_train, y_train))
# print(mlp.score(X_test, y_test))

# X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
#
# # mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10, 10], activation='tanh')
# mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10, 10])
# mlp.fit(X_train, y_train)
#
# print(mlp.score(X_train, y_train))
# print(mlp.score(X_test, y_test))


import matplotlib.pyplot as plt
import mglearn
#
# fig, axes = plt.subplots(2, 4, figsize=(20, 20))
# for axx, n_hidden_nodes in zip(axes, [10, 100]):
#     for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
#         mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes], alpha=alpha)
#         mlp.fit(X_train, y_train)
#         mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
#         mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
#         ax.set_title("n_hidden=[{}, {}]\nalpha={:.4f}".format(n_hidden_nodes, n_hidden_nodes, alpha))
# plt.show()
#


# fig, axes = plt.subplots(2, 4, figsize=(20, 8))
# for i, ax in enumerate(axes.ravel()):
#     mlp = MLPClassifier(solver='lbfgs', random_state=i, hidden_layer_sizes=[100, 100])
#     mlp.fit(X_train, y_train)
#     mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
#     mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
# plt.show()

from sklearn.datasets import load_breast_cancer

bcancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(bcancer.data, bcancer.target, random_state=0)
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)
print(mlp.score(X_train, y_train))
print(mlp.score(X_test, y_test))

mean_on_train = X_train.mean(axis=0)
std_on_train = X_train.std(axis=0)
X_train_scaled = (X_train - mean_on_train) / std_on_train
X_test_scaled = (X_test - mean_on_train) / std_on_train

# mlp = MLPClassifier(random_state=0)
mlp = MLPClassifier(random_state=0, max_iter=10000, alpha=1)
mlp.fit(X_train_scaled, y_train)

print(mlp.score(X_train_scaled, y_train))
print(mlp.score(X_test_scaled, y_test))





