import os
import numpy as np
from testutils import run_classification_experiment, get_random_acc

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

data = data[:int(0.4*len(data))]
targets = digits.target[:int(0.4*len(data))]

print(data.shape)
print(targets.shape)