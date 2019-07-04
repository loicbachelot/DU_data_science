import numpy as np
import sklearn
import sklearn.neighbors

"""
Prototype selection: condensed nearest neighbors
"""


class CNN(object):

    def __init__(self, data, labels, n_class=3, k=5):
        self.data = np.asarray(data)
        self.labels = np.asarray(labels)
        self.n_class = n_class
        self.k = k
        self.reduced_data, self.reduced_labels = self.init_reduced()

    def init_reduced(self):
        reduced_labels = []
        reduced_data = []

        for i in range(self.n_class):
            proba = np.isin(self.labels, i)
            proba = proba / proba.sum()
            index = np.random.choice(range(len(self.data)), p=proba)
            reduced_data.append(self.data[i])
            reduced_labels.append(i)

        return reduced_data, reduced_labels

    def run(self):

        cont = True
        while cont:
            cont = False
            knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=min(len(self.reduced_labels), self.k),
                                                         weights='distance')
            knn.fit(self.reduced_data, self.reduced_labels)
            for i in zip(self.data, self.labels):
                prediction = knn.predict([i[0]])
                if prediction[0] != i[1]:
                    cont = True
                    self.reduced_labels.append(i[1])
                    self.reduced_data.append(i[0])

        return self.reduced_data, self.reduced_labels


"""
Prototype selection: edited nearest neighbors
"""


class ENN(object):
    def __init__(self, data, labels, n_class=3, k=5):
        self.data = np.asarray(data)
        self.labels = np.asarray(labels)
        self.n_class = n_class
        self.k = k
        self.reduced_data, self.reduced_labels = data, labels

    def run(self):
        cont = True
        while cont:
            cont = False
            knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=self.k, weights='uniform')
            knn.fit(self.reduced_data, self.reduced_labels)
            predictions = knn.predict(self.reduced_data)

            remove_list = []
            for i in range(len(predictions)):
                if predictions[i] != self.reduced_labels[i]:
                    remove_list.append(i)
                    cont = True

            self.reduced_labels = np.delete(self.reduced_labels, remove_list)
            self.reduced_data = np.delete(self.reduced_data, remove_list, axis=0)

        return self.reduced_data, self.reduced_labels