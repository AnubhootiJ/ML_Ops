
from quiz_test_case import get_model, get_test_case, accuracy_class

def test_digit_correct_0():
    svm_clf, dec_clf = get_model()
    data = get_test_case(0)
    svm_pred = svm_clf.predict(data)
    dec_pred = dec_clf.predict(data)

    assert svm_pred == 0
    assert dec_pred == 0

def test_digit_correct_1():
    svm_clf, dec_clf = get_model()
    data = get_test_case(1)
    svm_pred = svm_clf.predict(data)
    dec_pred = dec_clf.predict(data)

    assert svm_pred == 1
    assert dec_pred == 1

def test_digit_correct_2():
    svm_clf, dec_clf = get_model()
    data = get_test_case(2)
    svm_pred = svm_clf.predict(data)
    dec_pred = dec_clf.predict(data)

    assert svm_pred == 2
    assert dec_pred == 2

def test_digit_correct_3():
    svm_clf, dec_clf = get_model()
    data = get_test_case(3)
    svm_pred = svm_clf.predict(data)
    dec_pred = dec_clf.predict(data)

    assert svm_pred == 3
    assert dec_pred == 3

def test_digit_correct_4():
    svm_clf, dec_clf = get_model()
    data = get_test_case(4)
    svm_pred = svm_clf.predict(data)
    dec_pred = dec_clf.predict(data)

    assert svm_pred == 4
    assert dec_pred == 4

def test_digit_correct_5():
    svm_clf, dec_clf = get_model()
    data = get_test_case(5)
    svm_pred = svm_clf.predict(data)
    dec_pred = dec_clf.predict(data)

    assert svm_pred == 5
    assert dec_pred == 5

def test_digit_correct_6():
    svm_clf, dec_clf = get_model()
    data = get_test_case(6)
    svm_pred = svm_clf.predict(data)
    dec_pred = dec_clf.predict(data)

    assert svm_pred == 6
    assert dec_pred == 6

def test_digit_correct_7():
    svm_clf, dec_clf = get_model()
    data = get_test_case(7)
    svm_pred = svm_clf.predict(data)
    dec_pred = dec_clf.predict(data)

    assert svm_pred == 7
    assert dec_pred == 7

def test_digit_correct_8():
    svm_clf, dec_clf = get_model()
    data = get_test_case(8)
    svm_pred = svm_clf.predict(data)
    dec_pred = dec_clf.predict(data)

    assert svm_pred == 8
    assert dec_pred == 8

def test_digit_correct_9():
    svm_clf, dec_clf = get_model()
    data = get_test_case(9)
    svm_pred = svm_clf.predict(data)
    dec_pred = dec_clf.predict(data)

    assert svm_pred == 9
    assert dec_pred == 9
        
def test_accuracy():
    svm_clf, dec_clf = get_model()
    svm_acc, dec_acc = accuracy_class(svm_clf, dec_clf, 0)
    svm_acc1, dec_acc1 = accuracy_class(svm_clf, dec_clf, 1)
    svm_acc2, dec_acc2 = accuracy_class(svm_clf, dec_clf, 2)
    svm_acc3, dec_acc3 = accuracy_class(svm_clf, dec_clf, 3)
    svm_acc4, dec_acc4 = accuracy_class(svm_clf, dec_clf, 4)
    svm_acc5, dec_acc5 = accuracy_class(svm_clf, dec_clf, 5)
    svm_acc6, dec_acc6 = accuracy_class(svm_clf, dec_clf, 6)
    svm_acc7, dec_acc7 = accuracy_class(svm_clf, dec_clf, 7)
    svm_acc8, dec_acc8 = accuracy_class(svm_clf, dec_clf, 8)
    svm_acc9, dec_acc9 = accuracy_class(svm_clf, dec_clf, 9)


    assert svm_acc > 0.8
    assert dec_acc > 0.8

    assert svm_acc1 > 0.8
    assert dec_acc1 > 0.8

    assert svm_acc2 > 0.8
    assert dec_acc2 > 0.8

    assert svm_acc3 > 0.8
    assert dec_acc3 > 0.8

    assert svm_acc4 > 0.8
    assert dec_acc4 > 0.8

    assert svm_acc5 > 0.8
    assert dec_acc5 > 0.8

    assert svm_acc6 > 0.8
    assert dec_acc6 > 0.8

    assert svm_acc7 > 0.8
    assert dec_acc7 > 0.8

    assert svm_acc8 > 0.8
    assert dec_acc8 > 0.8

    assert svm_acc9 > 0.8
    assert dec_acc9 > 0.8
    