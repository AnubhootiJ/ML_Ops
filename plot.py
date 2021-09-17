#Working virtual env - ops

## NEW THINGS - 
# Added Validation Set
# Storing and loading Models

import os
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from joblib import dump, load

digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

gammas = [1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05]
accuracy = []
print("=============================\nClassifying Handwritten Digits")
print("=============================")
"""
for gamma in gammas:
    #print("Working with gamma", gamma)
    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=gamma)

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    #predicted = clf.predict(X_test)
    ac = clf.score(X_test, y_test)
    #print("Accuracy = ", ac)
    accuracy.append(ac)

for i, gamma in enumerate(gammas):
    print(gamma, " = ", accuracy[i])

print("Plotting the Gamma Vs. Accuracy Plot")
plt.figure()
nran = [i+1 for i in range(6)]
plt.plot(nran, accuracy)
plt.title("Gamma Vs. Accuracy")
plt.xlabel("Gamma values")
plt.ylabel("Accuracy")
plt.xticks(nran, gammas)
plt.show()
"""

print("=============================\nCreating Validation Split")
print("=============================")

## We now create a split 70:15:!5, for train:val:test
x_train, x_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.3, shuffle=False)

x_val, x_test, y_val, y_test = train_test_split(
    x_test,y_test, test_size=0.5, shuffle=False)
print("Number of samples in train:val:test = {}:{}:{}".format(len(x_train), len(x_val), len(x_test)))

print("Now training...")
print("\nGamma\tTrain Acc Val Acc")
curr = os.getcwd()
candidates = []
for gamma in gammas:
    clf = svm.SVC(gamma=gamma)
    clf.fit(x_train, y_train)

    # Predict the value of the digit on the test subset
    #predicted = clf.predict(X_test)
    t_ac = clf.score(x_train, y_train)
    #ac = clf.score(x_test, y_test)
    val_ac = clf.score(x_val, y_val)
    
    #print("{} = {:.2f},\t {:.2f},\t {:.2f}".format(gamma, t_ac, val_ac, ac))
    print("{} = {:.2f},\t {:.2f}".format(gamma, t_ac, val_ac))

    if val_ac < 0.5:
        print("Skipping gamma {} because of low validation accuracy".format(gamma))
        continue

    cand = {
        'val' : val_ac,
        'train' : t_ac,
        'gamma' : gamma,
        #'model' : clf
    }
    """
    if b_ac < val_ac:
        best_clf = clf
        best_gamma = gamma
        b_ac = val_ac
        b_train = t_ac
        #b_test = ac
    """
    
    op_folder = curr + '/models/model_{}.joblib'.format(gamma)
    dump(clf, op_folder)
    candidates.append(cand)

best_cand = max(candidates, key=lambda x: x['val'])  
gamma = best_cand['gamma']     
op = curr+'/models/model_{}.joblib'.format(gamma)
mod = load(op)
print("\nBest Gamma Value = {} with a validation accuracy of {:.3f}".format(
    best_cand['gamma'], best_cand['val']))
#ac = best_cand['model'].score(x_test, y_test)
ac = mod.score(x_test, y_test)
print("Test accuracy for best gamma", ac)

## Model Selection to be done 
## store all instance of models to accomodate paralled training

