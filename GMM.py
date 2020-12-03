import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

sns.set()
import numpy as np
from sklearn.mixture import GaussianMixture
import time
import sklearn as sk
import pandas as pd

from sklearn.metrics import accuracy_score

# data_path = "D:\github\IDS2017-v4.1.0\kitsune_20_30w\\20_30w.tsv.npy"
# label_path = "D:\github\IDS2017-v4.1.0\kitsune_20_30w\labels_20-30w.csv.npy"
data_path = "./data/mawilab_6_10w.npy"
label_path = "./data/mawilab_label_10w.npy"

print("loading data")
data = np.load(data_path)
LEN = len(data)

labels = np.load(label_path)

from sklearn.preprocessing import StandardScaler

s1 = StandardScaler()
s1.fit(data)
data = s1.transform(data)
# data = data.astype(np.float32)
# labels = labels.astype(np.int64)

train_size = 90000
test_size = 10000

begin = time.time()
data_train = data[0:train_size, :]
data_test = data[train_size:train_size + test_size, :]
label_train = labels[0:train_size]
label_test = labels[train_size:train_size + test_size]

# all_indices = np.random.permutation(train_size + test_size)  # random
# data_train = data[all_indices[0:train_size], :]
# data_test = data[all_indices[train_size:train_size + test_size], :]
# label_train = labels[all_indices[0:train_size]]
# label_test = labels[all_indices[train_size:train_size + test_size]]
print("loading done")

gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
gmm.fit(data_train)

print("train done")
pre = gmm.predict(data_test)
print("predict done")
# accuracy = accuracy_score(label_test,pre)
# print(accuracy)
end = time.time()
print("time is ", end - begin)

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

if TN + FP != 0:
    fpr = FP / (TN + FP)
else:
    fpr = 0
print("fpr = ", fpr)
# print("-------------")
#
# newPrecision = sk.metrics.precision_score(label_test, pre)
# newRecall = sk.metrics.recall_score(label_test, pre)
# newf1 = sk.metrics.f1_score(label_test, pre)
#
# print("Precision ", newPrecision, "Recall", newRecall, "F1-score", newf1)
# PRECISION = []
# RECALL = []
# F1 = []
# PRECISION.append(newPrecision)
# RECALL.append(newRecall)
# F1.append(newf1)
#
# end = time.time()
# print("time is ", end - begin)
#
# out_path = "E:\半监督\结果对比\Kitsune特征\GMM\\10000"
# df1 = pd.DataFrame(data=PRECISION)
# df2 = pd.DataFrame(data=RECALL)
# df3 = pd.DataFrame(data=F1)
# df1.to_csv(out_path + "\precision.csv")
# df2.to_csv(out_path + "\\recall.csv")
# df3.to_csv(out_path + "\F1.csv")
