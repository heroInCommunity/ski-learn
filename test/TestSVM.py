import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_olivetti_faces
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score, KFold
from sklearn import preprocessing

from scipy.stats import sem
from sklearn import metrics
import csv
import glob
from PIL import Image
import matplotlib.image as mpimg

csvData = csv.reader(open('D:\gmm\ML\CIFAR\TrainLabels.csv'), delimiter = ',')
csvList = list(csvData)
del csvList[0]

STOP = 12000#len(csvList)

targets = np.empty(STOP, dtype = str)
counter = 0
for row in csvList:
    if (counter == STOP): break
    targets[int(row[0]) - 1] = row[1]
    counter += 1

data = np.empty((STOP, 32 * 32), dtype = list)

path = 'D:/gmm/ML/CIFAR/train/*.png'   
files = glob.glob(path)

placeholder = 'D:/gmm/ML/CIFAR/train/'

for i in range(1, STOP + 1):    
    file = placeholder + str(i) + '.png'
    pic = Image.open(file).convert('L')
#     pic = mpimg.imread(file)
    
    pix = np.array(pic)        
    data[i - 1] = pix.ravel()

# print(data.shape)
      

# faces = fetch_olivetti_faces()
# print(faces.DESCR)

# print(faces.keys())
# print(faces.images.shape)
# print(faces.data.shape)
# print(faces.target.shape)

def print_faces(images, target, top_n):
    # set up the figure size in inches
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(top_n):
        # plot the images in a matrix of 20x20
        p = fig.add_subplot(20, 20, i + 1, xticks=[],
        yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)
        # label the image with the target value
        p.text(0, 14, str(target[i]))
        p.text(0, 60, str(i))
    plt.show()
    
def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold croos validation iterator
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print(("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores)))

def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    
    print("Accuracy on training set:")
    print(clf.score(X_train, y_train))
    
    print("Accuracy on testing set:")
    print(clf.score(X_test, y_test))
    
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))

# print_faces(faces.images, faces.target, 20)

X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.25, random_state=0)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

svc_1 = SVC(kernel='linear')
# evaluate_cross_validation(svc_1, X_train, y_train, 5)

# for j in range(0, 1024):
#     print(X_train[0][j])
#     if data[0][j] == None:
#         print(j)
#         break
# print(data.shape)
train_and_evaluate(svc_1, X_train, X_test, y_train, y_test)
























