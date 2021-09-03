import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

gammas = [1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05]
accuracy = []
for gamma in gammas:
    print("Working with gamma", gamma)
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
    print("Accuracy = ", ac)
    accuracy.append(ac)

#print(accuracy)

plt.figure()
nran = [i+1 for i in range(6)]
plt.plot(nran, accuracy)
plt.title("Gamma Vs. Accuracy")
plt.xlabel("Gamma values")
plt.ylabel("Accuracy")
plt.xticks(nran, gammas)
plt.show()