from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

import pselection

iris = datasets.load_iris()

cnn = pselection.CNN(iris["data"], iris["target"])
enn = pselection.ENN(iris.data, iris.target)

reduced_data_cnn, reduced_labels_cnn = cnn.run()
reduced_data_enn, reduced_labels_enn = enn.run()
print(len(iris.target), len(reduced_labels_cnn))
print(len(iris.target), len(reduced_labels_enn))
