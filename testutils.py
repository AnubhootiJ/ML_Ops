import os
import numpy as np
from joblib import dump, load
from sklearn import datasets, svm, metrics

def test_model(best_model_path, X, y):
    clf = load(best_model_path)
    metrics = test(clf, X, y)
    return metrics

def test(clf, X, y):
    predicted = clf.predict(X)
    acc = metrics.accuracy_score(y_pred=predicted, y_true=y)
    f1 = metrics.f1_score(y_pred=predicted,y_true=y, average='macro')

    return {"acc":acc, "f1":f1}

def get_random_acc(y):
    return max(np.bincount(y))/len(y)

def run_classification_experiment(
    clf, X_train, y_train, 
    X_valid, y_valid, gamma,
    output_model_file, skip_dummy=True):

    random_val_acc = get_random_acc(y_valid)
    #clf = classifier(gamma=gamma)
    clf.fit(X_train, y_train)
    metrics_value = test(clf, X_valid, y_valid)

    if skip_dummy and metrics_value['acc'] < random_val_acc:
        print("skipping for {}".format(gamma))
        return None

    output_folder = os.path.dirname(output_model_file)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    dump(clf, output_model_file)
    return metrics_value
