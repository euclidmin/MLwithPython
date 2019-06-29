import mglearn.plots
# import mglearn.plot_animal_tree

# mglearn.plots.plot_animal_tree()


from sklearn.datasets import load_breast_cancer
from sklearn.datasets import train_test_split
from sklearn.tree import DecisionTreeClassifier

cancer = load_breast_cancer()
X_train, Xtest, y_train, y_test = train_test_split(cancer.data, )
