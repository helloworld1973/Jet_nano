import PIL
import numpy as np
from sklearn import svm
from joblib import dump, load
import os
from os import listdir
from os.path import isfile, join
import torchvision.transforms as transforms
from skimage.feature import hog
from PIL import Image
from sklearn.model_selection import train_test_split


# read data
def get_label(path):
    return int(path.split("_")[0])

mypath = 'E:\\Python Projects\\Jet_nano\\dataset_xy_SVM\\'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
X = []
y = []
for file_name in onlyfiles:
    a_label = file_name.split('_')[0]
    a_image = PIL.Image.open(mypath + file_name)
    a_image = transforms.functional.resize(a_image, (32, 32))
    a_image = a_image.convert('RGBA')
    arr = np.array(a_image)[:, :, 0:3]

    flat_arr = arr.ravel().tolist()
    X.append(flat_arr)
    #
    #hog_features = hog(arr, orientations=8, pixels_per_cell=(4, 4),cells_per_block=(1, 1), visualize=False)
    #X.append(hog_features)
    y.append(a_label)
    # print()

X = np.asarray(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

clf = svm.SVC()
# clf = svm.SVC(C=1, probability=True)
clf.fit(x_train, y_train)

s = dump(clf, 'SVM_classify.joblib')
clf2 = load('SVM_classify.joblib')

a = clf2.score(x_test, y_test)
print("Accuracy of the SVM Classifier is: {}%".format(round(a, 2)))


