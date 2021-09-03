### QUIZ-1
## Tasks to perform -
# 3 different resolutions 
# Play around with train-test split
# Submission: create new feature branch feature/quiz1

import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.transform import resize

digits = datasets.load_digits()

n_samples = len(digits.images)
print("Number of samples in the dataset =", n_samples)
print("Actual Resolution =", digits.images.shape) # 1797 * 8 * 8
data = digits.images.reshape((n_samples, -1))
print("Flattened Resolution =", data.shape)

# Different Resolution - 4*4, 6*6, 8*8
# Different splits - 70:30, 80:20, 90:10

#data4 = np.resize(digits.images, (1797, 4,4))
#print(data4.shape)

resolution = [4,6,8]
splits = [0.7,0.8,0.9]
accuracy = []
f1sc = []
for res in resolution:
    for split in splits:
        """
        if res !=8:
            data = np.resize(digits.images, (1797, res,res))
            data = data.reshape((n_samples, -1)) 
        else:
            data = digits.images.reshape((n_samples, -1))
        #print(data.shape)
        """
        data = resize(digits.images, (1797, res, res))
        data = data.reshape((n_samples, -1)) 
        clf = svm.SVC(gamma=0.001)

        # Split data into 50% train and 50% test subsets
        X_train, X_test, y_train, y_test = train_test_split(
            data, digits.target, test_size=1-split, shuffle=False)
        #print(len(X_train))

        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # Predict the value of the digit on the test subset
        predicted = clf.predict(X_test)
        ac = clf.score(X_test, y_test)
        #print("Accuracy for split", split," and resolution", res, "= ", ac)
        accuracy.append(ac)
        f1 = metrics.f1_score(y_test, predicted, average='weighted')
        #print("F1",f1)
        f1sc.append(f1)

print("##### ACCURACY METRICS ########")
print("Resolution\tSplit\tAccuracy\tF1-Score")
i = 0 
for res in resolution:
    for split in splits:
        if split == 0.7:
            sp = "70:30"
        elif split == 0.8:
            sp = "80:20"
        else:
            sp="90:10"
        if res == 4:
            re = "4*4"
        elif res == 6:
            re = "6*6"
        else:
            re = "8*8"
        print(re,"\t\t",sp,"\t",accuracy[i],"\t",f1sc[i])
        i+=1 
        


        

