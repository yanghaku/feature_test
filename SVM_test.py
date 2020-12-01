from sklearn import svm
import pandas as pd
import numpy as np
import time
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

data_path = "./data/mawilab_10w.npy"
label_path = "./data/mawilab_label_10w.npy"

print("data loading")
data = np.load(data_path)
labels = np.load(label_path)

train_size = 90000
test_size = 10000

from sklearn.preprocessing import StandardScaler

s1 = StandardScaler()
s1.fit(data)
data = s1.transform(data)
data = data.astype(np.float32)
labels = labels.astype(np.int64)

begin = time.time()
data_train_all = data[0:train_size, :]
data_test_all = data[train_size:train_size + test_size, :]
label_train = labels[0:train_size]
label_test = labels[train_size:train_size + test_size]
print("loading done")
print(label_train.shape)
print(data_train_all.shape)
slides = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
          [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
          [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
          [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
          [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74],
          [75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
          [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
           103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
           116, 117, 118, 119, 120, 121, 122, 123, 124],
          [125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
           138, 139],
          [140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152,
           153, 154],
          [155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
           168, 169],
          [170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182,
           183, 184],
          [185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
           198, 199],
          [200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212,
           213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225,
           226, 227, 228, 229, 230, 231, 232, 233, 234]
          ]

F1s = []
Precision = []
Recall = []
FPR = []

for i in range(len(slides)):
    data_train = data_train_all[:, slides[i]]
    data_test = data_test_all[:, slides[i]]
    clf = svm.SVC()
    print("training")
    clf.fit(data_train, label_train)
    print("training done")

    print("texting")
    pre = clf.predict(data_test)

    print("texting done")

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
    # P = TP / (TP + FP)
    # R = TP / (TP + FN)
    # F1 = 2 * P * R / (P + R)
    # print("TPR = ", TP / (TP + FN), "FPR = ", FP / (TN + FP), "Precision", TP / (TP + FP), "F1-Score", F1)

    accuracy = accuracy_score(label_test, pre)
    print(accuracy)
    end = time.time()
    print("time is ", end - begin)
    F1s.append(f1_score(label_test, pre))
    Precision.append(precision_score(label_test, pre))
    Recall.append(recall_score(label_test, pre))

    if TN + FP != 0:
        fpr = FP / (TN + FP)
    else:
        fpr = 0
    print("fpr = ", fpr)
    FPR.append(fpr)
