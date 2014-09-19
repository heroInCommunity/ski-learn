import sklearn as sk
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.pipeline import Pipeline

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

from scipy.stats import sem

def mean_score(scores):
    return ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))

iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target
X, y = X_iris[:, :2], y_iris

# create a composite estimator made by a pipeline of the standarization and the linear model
clf = Pipeline([
('scaler', preprocessing.StandardScaler()),
('linear_model', SGDClassifier())
])
# create a k-fold cross validation iterator of k=5 folds
cv = KFold(X.shape[0], 5, shuffle=True, random_state=33)
# by default the score used is the one returned by score method of the estimator (accuracy)
scores = cross_val_score(clf, X, y, cv=cv)
print(scores)

print(mean_score(scores))