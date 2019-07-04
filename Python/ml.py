import pandas as pd
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import tree
from sklearn.linear_model import Perceptron
import pselection
import os.path
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error


def main():
    iris = datasets.load_iris()
    perceptron(iris)
    perceptronPS(iris)
    kmeans(iris)
    kmeansPS(iris)
    decisionTree(iris)
    decisionTreePS(iris)
    return 0


def kmeansPS(iris):
    cnn = pselection.CNN(iris["data"], iris["target"])
    enn = pselection.ENN(iris.data, iris.target)
    reduced_data_cnn, reduced_labels_cnn = cnn.run()
    reduced_data_enn, reduced_labels_enn = enn.run()

    # PS CNN
    x = pd.DataFrame(reduced_data_cnn)
    x.columns = ['Sepal_Length', 'Sepal_width', 'Petal_Length', 'Petal_width']
    y = pd.DataFrame(reduced_labels_cnn)
    y.columns = ['Targets']
    model = KMeans(n_clusters=3)
    model.fit(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    y_pred = model.predict(X_test)
    print("Kmeans with prototype selection CNN:")
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print('Recall %.2f' % recall_score(y_test, y_pred, average='micro'))
    print('Error %.2f' % mean_squared_error(y_test, y_pred))

    # PS ENN
    x = pd.DataFrame(reduced_data_enn)
    x.columns = ['Sepal_Length', 'Sepal_width', 'Petal_Length', 'Petal_width']
    y = pd.DataFrame(reduced_labels_enn)
    y.columns = ['Targets']
    model = KMeans(n_clusters=3)
    model.fit(x)
    y_pred = model.predict(X_test)
    print("Kmeans with prototype selection ENN:")
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print('Recall %.2f' % recall_score(y_test, y_pred, average='micro'))
    print('Error %.2f' % mean_squared_error(y_test, y_pred))


def kmeans(iris):
    x = pd.DataFrame(iris.data)
    x.columns = ['Sepal_Length', 'Sepal_width', 'Petal_Length', 'Petal_width']

    y = pd.DataFrame(iris.target)
    y.columns = ['Targets']

    model = KMeans(n_clusters=3)
    model.fit(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    y_pred = model.predict(X_test)

    print("Kmeans with no prototype selection:")
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print('Recall %.2f' % recall_score(y_test, y_pred, average='micro'))
    print('Error %.2f' % mean_squared_error(y_test, y_pred))


def perceptron(iris):
    x = pd.DataFrame(iris.data)
    x.columns = ['Sepal_Length', 'Sepal_width', 'Petal_Length', 'Petal_width']

    y = pd.DataFrame(iris.target)
    y.columns = ['Targets']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
    ppn.fit(X_train, y_train)
    y_pred = ppn.predict(X_test)
    from sklearn.metrics import accuracy_score
    print("perceptron with no prototype selection:")
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print('Recall %.2f' % recall_score(y_test, y_pred, average='micro'))
    print('Error %.2f' % mean_squared_error(y_test, y_pred))


def perceptronPS(iris):
    cnn = pselection.CNN(iris["data"], iris["target"])
    enn = pselection.ENN(iris.data, iris.target)
    reduced_data_cnn, reduced_labels_cnn = cnn.run()
    reduced_data_enn, reduced_labels_enn = enn.run()

    # PS with CNN
    x = pd.DataFrame(reduced_data_cnn)
    x.columns = ['Sepal_Length', 'Sepal_width', 'Petal_Length', 'Petal_width']
    y = pd.DataFrame(reduced_labels_cnn)
    y.columns = ['Targets']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
    ppn.fit(X_train, y_train)
    y_pred = ppn.predict(X_test)
    print("perceptron with prototype selection CNN:")
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print('Recall %.2f' % recall_score(y_test, y_pred, average='micro'))
    print('Error %.2f' % mean_squared_error(y_test, y_pred))

    # PS with ENN
    x = pd.DataFrame(reduced_data_enn)
    x.columns = ['Sepal_Length', 'Sepal_width', 'Petal_Length', 'Petal_width']
    y = pd.DataFrame(reduced_labels_enn)
    y.columns = ['Targets']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
    ppn.fit(X_train, y_train)
    y_pred = ppn.predict(X_test)
    print("perceptron with prototype selection ENN:")
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print('Recall %.2f' % recall_score(y_test, y_pred, average='micro'))
    print('Error %.2f' % mean_squared_error(y_test, y_pred))

def decisionTree(iris):
    x = pd.DataFrame(iris.data)
    x.columns = ['Sepal_Length', 'Sepal_width', 'Petal_Length', 'Petal_width']

    y = pd.DataFrame(iris.target)
    y.columns = ['Targets']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(iris.data, iris.target)
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    print("decision tree with no prototype selection:")
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print('Recall %.2f' % recall_score(y_test, y_pred, average='micro'))
    print('Error %.2f' % mean_squared_error(y_test, y_pred))


def decisionTreePS(iris):
    cnn = pselection.CNN(iris["data"], iris["target"])
    enn = pselection.ENN(iris.data, iris.target)
    reduced_data_cnn, reduced_labels_cnn = cnn.run()
    reduced_data_enn, reduced_labels_enn = enn.run()

    # PS with CNN
    x = pd.DataFrame(reduced_data_cnn)
    x.columns = ['Sepal_Length', 'Sepal_width', 'Petal_Length', 'Petal_width']
    y = pd.DataFrame(reduced_labels_cnn)
    y.columns = ['Targets']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(iris.data, iris.target)
    y_pred = clf.predict(X_test)
    print("decision tree with prototype selection CNN:")
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print('Recall %.2f' % recall_score(y_test, y_pred, average='micro'))
    print('Error %.2f' % mean_squared_error(y_test, y_pred))

    # PS with ENN
    x = pd.DataFrame(reduced_data_enn)
    x.columns = ['Sepal_Length', 'Sepal_width', 'Petal_Length', 'Petal_width']
    y = pd.DataFrame(reduced_labels_enn)
    y.columns = ['Targets']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(iris.data, iris.target)
    y_pred = clf.predict(X_test)
    print("decision tree with prototype selection ENN:")
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print('Recall %.2f' % recall_score(y_test, y_pred, average='micro'))
    print('Error %.2f' % mean_squared_error(y_test, y_pred))



if __name__ == '__main__':
    main_path = sys.argv[0]
    if (os.path.isfile(os.path.abspath(main_path))):
        os.chdir(os.path.dirname(os.path.abspath(main_path)))
    status = main()
    sys.exit(status)
