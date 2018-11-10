from sklearn.svm import SVC
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from collections import Counter


class KNNClassifier(object):
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def euclideanDist(self, a, b):
        '''
        this method takess advantage of the linalg.norm function included in numpy.
        this function performs better than a for loop written in Python to calculate the euclidean distance

        :return: the euclidean distance between point a and b
        '''

        return np.linalg.norm(a-b)

    def closest(self, row, k):
        '''
        :param row: a row in the dataset
        :param k: number of neighbors to collect
        :return: 0 or 1 depending on the the majority vote collected

        first gets the distances of all the points from the sameple, then computes their indices of the smallest
        distances between the target point
        and adds them to an array. For each index, the true class label is collected and appended to an array.
        The values in the last array are checked to compute the majority vote.
        '''
        dist = [self.euclideanDist(row, trainer) for trainer in self.X_train]

        best_dist = []

        for i in range(k):
            min_index = dist.index(min(dist))
            best_dist.append(min_index)
            dist.remove(min(dist))

        labels = []

        for i in range(len(best_dist)):
            labels.append(self.y_train[best_dist[i]])

        class0 = 0
        class1 = 0

        for i in range(len(labels)):
            if labels[i] == 0:
                class0 += 1
            else:
                class1 += 1

        if class0 > class1:
            return 0
        else:
            return 1

    def fit(self, training_data, labels):
        '''
        Note: there is really no fitting happening here like any other ML algorithm.
        Rather, this is my way of setting the training_data and labels before prediction time.
        I also wanted my implementation to mimic the sklearn one
        '''
        self.X_train = training_data
        self.y_train = labels

    def predict(self, test_data):
        predictions = []
        for row in test_data:
            label = self.closest(row, 1)
            predictions.append(label)

        return predictions

# ----------------- end class declaration -----------------


# read in the data
data = pd.read_csv("data.csv")

# drop id as it has no use in this analysis
data = data.drop("id", axis=1)

# change diagnosis from type char to type int
for index, row in data.iterrows():
    if row["diagnosis"] == "B":
        data.loc[index, "diagnosis"] = 0
    else:
        data.loc[index, "diagnosis"] = 1

features_mean=list(data.columns[1:11])

malignant = data[data["diagnosis"] == 1]
benign = data[data["diagnosis"] == 0]

plt.rcParams.update({'font.size': 8})
fig, axes = plt.subplots(nrows=5, ncols=2)
axes = axes.ravel()

for idx, ax in enumerate(axes):
    ax.figure
    binwidth = (max(data[features_mean[idx]]) - min(data[features_mean[idx]])) / 50
    ax.hist([malignant[features_mean[idx]], benign[features_mean[idx]]],
            bins=np.arange(min(data[features_mean[idx]]), max(data[features_mean[idx]]) + binwidth, binwidth), alpha=0.5,
            stacked=True, density=True, label=['M', 'B'], color=['r', 'b'])
    ax.legend(loc='upper right')
    ax.set_title(features_mean[idx])
plt.tight_layout()
#plt.savefig('./feature means', dpi=300)
plt.show()


# extract true class labels from features
X, y = data.iloc[:, 1:].values, data.iloc[:, 0].values

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=0.5, random_state=0)

# standardize training data
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
X_dev_std = stdsc.transform(X_dev)

# create SVM object and fit to the training dataset
svm = SVC(kernel='rbf', C=5.0, gamma='auto')
# svm = SVC(kernel='linear, C=1.0, gamma='auto')
# svm = SVC(C=5.0, gamma='auto')
svm.fit(X_train_std, y_train)

# create KNN object and predict outcomes
knn = KNNClassifier()
knn.fit(X_train_std, y_train)
# for the dev set
predictions_dev = knn.predict(X_dev_std)
score_dev = metrics.accuracy_score(y_dev, predictions_dev)
# for the test set
predictions_test = knn.predict(X_test_std)
score_test = metrics.accuracy_score(y_test, predictions_test)


print("score for SVM with best settings on dev set:\n {0:f}".format((svm.score(X_dev_std, y_dev))*100))
print("score for KNN implementation on dev set:\n {0:f}".format(score_dev * 100))

print("score for SVM with best settings on test set:\n {0:f}".format((svm.score(X_test_std, y_test))*100))
print("score for KNN implementation on test set:\n {0:f}".format(score_test * 100))
