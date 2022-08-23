from skimage.feature import hog
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from glob import glob
import time
from sklearn.metrics import classification_report

image_paths = []
y = []

for class_path in glob("./traffic-signs-classification/myData/*"):

    cpath_replaced = class_path.replace("./traffic-signs-classification/myData\\", "")

    i = cpath_replaced[0]
    if len(cpath_replaced) > 1:
        if cpath_replaced[1] in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"):
            i = int(i + cpath_replaced[1])

    for image_path in glob(class_path + "/*"):
        image_paths.append(image_path)
        y.append(i)

for sample_label, sample_path in zip(y[:5], image_paths[:5]):
    print("Label: {} Path: {}".format(sample_label, sample_path))

IMAGE_SIZE = (32, 32)


# Our dataset is already resized so resize parameter's value is false default.
def read_image(img_path, resize=False):
    img = Image.open(img_path)
    if resize:
        img.resize(IMAGE_SIZE)
    return np.asarray(img)


x = np.asarray([read_image(image_path) for image_path in image_paths])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y)

x_train_fd = []
x_test_fd = []

for image in x_train:
    x_train_fd.append(image.ravel().tolist())
    # x_train_fd.append(hog(image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=False))

for image in x_test:
    x_test_fd.append(image.ravel().tolist())
    # x_test_fd.append(hog(image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=False))

x_train_fd = np.asarray(x_train_fd)
x_test_fd = np.asarray(x_test_fd)
print(x_train_fd.shape)
print(x_test_fd.shape)

# SVM
before_SVM_time = int(round(time.time() * 1000))
clf_SVM = SVC()
clf_SVM.fit(x_train_fd, y_train)
SVM_predicted_y = clf_SVM.predict(x_test_fd)
after_SVM_time = int(round(time.time() * 1000))

'''
# MLP
before_MLP_time = int(round(time.time() * 1000))
model = MLPClassifier(random_state=1, learning_rate_init=0.0001, max_iter=100000)
model.fit(x_train_fd, y_train)
print("Model Fitted")
MLP_predicted_y = model.predict(x_test_fd)
after_MLP_time = int(round(time.time() * 1000))

# Evaluate model performance
print('MLP:')
print('time:' + str(after_MLP_time - before_MLP_time))
print(classification_report(y_test, MLP_predicted_y))
print('SVM:')
print('time:' + str(after_SVM_time - before_SVM_time))
print(classification_report(y_test, SVM_predicted_y))
'''