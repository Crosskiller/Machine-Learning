
#Cross-validation page-37

from matplotlib import pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
data = load_iris()

features = data.data
features_names = data.feature_names
target = data.target
target_names = data.target_names

labels = target_names[target]
plength = features[:, 2]
is_setosa = (labels == 'setosa')

features = features[~is_setosa]
labels = labels[~is_setosa]
is_virginica = (labels=='virginica') #shape = 100
#print is_virginica.shape #shape = 75


def is_virginica_test(fi, t, reverse,example):
    test = example[:,fi]>t
    if reverse:
        test = not test
    return test


def fit_model(features1, is_virginica1, training1):
    best_acc = -1.0
    for fi in range(features1.shape[1]):
        thresh = features1[:,fi]
        for t in thresh:
            feature_i = features1[:, fi]
            pred = (feature_i > t)
            acc = (pred == is_virginica1).mean()
            rev_acc = (pred == ~is_virginica1).mean()
            if(rev_acc > acc):
                reverse = True
                acc = rev_acc
            else:
                reverse = False

            if acc > best_acc:
                best_acc = acc
                best_fi = fi
                best_t = t
                best_reverse = reverse
    return is_virginica_test(best_fi, best_t, best_reverse, training1)


correct = 0.0
for ei in range(len(features)):
    training = np.ones(len(features), bool)
    training[ei] = False
    testing = ~training
    prediction = fit_model(features[training], is_virginica[training], features[testing])
    correct += np.sum(prediction==is_virginica[testing])
    print correct
acc = correct/float(len(features))
print ('Accuracy: {0:.1%}'.format(acc))