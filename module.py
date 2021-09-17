#Working virtual env - ops

## NEW THINGS - 
# Modularizing 

import os
import matplotlib.pyplot as plt
from utils import preprocess, split_data, get_best, plot

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from joblib import dump, load

print("=============================\nClassifying Handwritten Digits")
print("=============================")

digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

gammas = [1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05]
splits = [(0.15,0.15), (0.20,0.10)]
resolution = [4, 8, 16, 32]
curr = os.getcwd()

print("Now training...")
candidates = []
i=1
fig = plt.figure()
for res in resolution:
    data = preprocess(digits.images, res)
    for split in splits:
        x_train, y_train, x_test, y_test, x_val, y_val = split_data(
            data, digits.target, split)
        vaccs = []
        taccs = []
        print("Res  Val:Test  Gamma  Train Acc  Val Acc")
        for gamma in gammas:

            clf = svm.SVC(gamma=gamma)
            clf.fit(x_train, y_train)
            t_ac = clf.score(x_train, y_train)
            val_ac = clf.score(x_val, y_val)

            #if val_ac < 0.5:
            #    print("Skipping gamma {} because of low validation accuracy".format(gamma))
            #    continue

            print("{}, {}, {} = {:.2f},\t {:.2f}".format(res, split, gamma, t_ac, val_ac))
            cand = {
                'val' : val_ac,
                'train' : t_ac,
                'gamma' : gamma,
                'split' : split,
                'res' : res
            }
            op_folder = curr + '/models/model_{}_{}_{}.joblib'.format(res, split, gamma)
            dump(clf, op_folder)
            candidates.append(cand)
            vaccs.append(val_ac)
            taccs.append(t_ac)
        ax = fig.add_subplot(4,2,i)
        plot(ax, vaccs, taccs, gammas, split, res)
        i+=1
        
plt.show()
best_cand = get_best(candidates, curr)



