from sklearn.svm import OneClassSVM
import pandas as pd
import numpy as np
import time
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# data_path = "data/mawilab_10w.npy"
# data_path = "D:\\Dataset\\mawilab_kitsune.npy"
# label_path = "data/mawilab_label_10w.npy"
data_path = "D:\\Dataset\\IDS2017-Wednesday\\IDS2017-v4.1.0\\data_30w_des.tsv.npy"
label_path = "D:\\Dataset\\IDS2017-Wednesday\\IDS2017-v4.1.0\\labels_30w_des.csv.npy"
print("data loading")
data = np.load(data_path)
labels = np.load(label_path)

# train_size = 90000
# test_size = 10000
train_size = 270000
test_size = 29999

begin = time.time()
data_train = data[0:train_size, :]
data_test = data[train_size:train_size + test_size, :]
label_train = labels[0:train_size]
label_test = labels[train_size:train_size + test_size]
print("loading done")
print(label_train.shape)
print(data_train.shape)

print("training")
clf = OneClassSVM(gamma='auto')

train = []
for i in range(train_size):
    if label_train[i] == 0:
        train.append(data_train[i])
clf.fit(np.array(train))

print("training done")

print("testing")
predd = clf.predict(data_test)

pre = []
for i in predd:
    if i==1:
        pre.append(0)
    else:
        pre.append(1)

print("testing done")

print("f1 score = ", f1_score(label_test, pre))
print("precision = ", precision_score(label_test, pre))
print("recall = ", recall_score(label_test, pre))

TP = 0
FP = 0
TN = 0
FN = 0
for i in range(len(pre)):
    if pre[i] == 0:
        if label_test[i] == 0:
            TN = TN + 1
        else:
            FN = FN + 1
    else:
        if label_test[i] == 1:
            TP = TP + 1
        else:
            FP = FP + 1

print("TP = ", TP, " FP = ", FP, " TN = ", TN, " FN = ", FN)
P = TP / (TP + FP)
R = TP / (TP + FN)
F1 = 2 * P * R / (P + R)
print("TPR = ", TP / (TP + FN), "FPR = ", FP / (TN + FP), "Precision", TP / (TP + FP), "F1-Score", F1)

accuracy = accuracy_score(label_test, pre)
print(accuracy)
end = time.time()
print("time is ", end - begin)
