
import os
import matplotlib.pyplot as plt
from utils import preprocess, split_data, get_best, plot

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load

print("=============================\nClassifying Handwritten Digits")
print("=============================")


digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

#
# gammas = [1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05]
splits = [(0.15,0.15), (0.20,0.10), (0.1, 0.1), (0.1, 0.05), (0.05, 0.05)]
resolution = [32]
curr = os.getcwd()

print("Now training...")
#candidates = []
i=1
#fig = plt.figure()
print("Model\t\tRes  Val:Test  Train Acc  Val Acc f1 Score")
for res in resolution:
    data = preprocess(digits.images, res)
    for split in splits:
        tsplit = split[0] + split[1]
        x_train, y_train, x_test, y_test, x_val, y_val = split_data(
            data, digits.target, split, tsplit)
        vaccs = []
        taccs = []

        SVMvaccs = []
        SVMtaccs = []
        
        #for gamma in gammas:

        dec = DecisionTreeClassifier(random_state = 0)
        dec.fit(x_train, y_train)
        t_ac = dec.score(x_train, y_train)
        val_ac = dec.score(x_val, y_val)
        predicted = dec.predict(x_test)
        f1 = metrics.f1_score(y_pred=predicted,y_true=y_test, average='macro')

        vaccs.append(val_ac)
        taccs.append(t_ac)

        #if val_ac < 0.5:
        #    print("Skipping gamma {} because of low validation accuracy".format(gamma))
        #    continue

        print("Dec tree: {}, {} = {:.2f},\t {:.2f},\t {:.2f}".format(res, split, t_ac, val_ac, f1))
        
        clf = svm.SVC(gamma=0.001)
        clf.fit(x_train, y_train)
        t_ac = clf.score(x_train, y_train)
        val_ac = clf.score(x_val, y_val)
        SVMvaccs.append(val_ac)
        SVMtaccs.append(t_ac)
        predicted = clf.predict(x_test)
        f1 = metrics.f1_score(y_pred=predicted,y_true=y_test, average='macro')

        print("SVM with g=0.5: {}, {} = {:.2f},\t {:.2f},\t {:.2f}".format(res, split, t_ac, val_ac, f1))
        
        """
        cand = {
            'val' : val_ac,
            'train' : t_ac,
            'split' : split,
            'res' : res
        }
        
        op_folder = curr + '/models/decisionTree/dec_{}_{}.joblib'.format(res, split)
        dump(dec, op_folder)
        candidates.append(cand)
        """
        
    #ax = fig.add_subplot(4,2,i)
    #plot(ax, vaccs, taccs, split, res)
    i+=1
        
#plt.show()
#best_cand = get_best(candidates, curr)



