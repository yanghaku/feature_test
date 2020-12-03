import seaborn as sns;sns.set()
import numpy as np
from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics

from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM

import time
from sklearn.metrics import accuracy_score
#
# data_path = "D:\github\IDS2017-v4.1.0\data_30w_des.tsv.npy"
# label_path = "D:\github\IDS2017-v4.1.0\labels_30w_des.csv.npy"

data_path = "./data/mawilab_6_10w.npy"
label_path = "./data/mawilab_label_10w.npy"

print("loading data")
data = np.load(data_path)
labels = np.load(label_path)
from sklearn.preprocessing import StandardScaler
s1 = StandardScaler()
s1.fit(data)
data = s1.transform(data)
# data = data.astype(np.float32)
# labels = labels.astype(np.int64)

# train_size = 280000
# test_size = 19999

train_size = 90000
test_size = 10000

begin = time.time()
# data_train = data[0:train_size,:]
# data_test = data[train_size:train_size+test_size,:]
# label_train = labels[0:train_size]
# label_test = labels[train_size:train_size+test_size]

# all_indices = np.random.permutation(train_size + test_size)  # random
# data_train = data[all_indices[0:train_size], :]
# data_test = data[all_indices[train_size:train_size + test_size], :]
# label_train = labels[all_indices[0:train_size]]
# label_test = labels[all_indices[train_size:train_size + test_size]]
data_train = data[0:train_size, :]
data_test = data[train_size:train_size + test_size, :]
label_train = labels[0:train_size]
label_test = labels[train_size:train_size + test_size]
print("loading done")


# RBM = BernoulliRBM()
# RBM.fit(data_train)
# pre = RBM.score_samples(data_test)
# for i in range(100):
#     print(pre[i])

logistic = linear_model.LogisticRegression(solver='newton-cg', tol=1)
rbm = BernoulliRBM(random_state=0,verbose=True)
classifier = Pipeline(steps=[('rbm',rbm),('logistic',logistic)])

rbm.learning_rate = 0.06
rbm.n_iter = 10
rbm.n_components = 100
logistic.C = 6000.0

classifier.fit(data_train,label_train)

pre = classifier.predict(data_test)
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(label_test, pre)))


print("train done")

print("predict done")
#accuracy = accuracy_score(label_test,pre)
#print(accuracy)
end = time.time()
print("time is ",end - begin)

TP = 0
FP = 0
TN = 0
FN = 0


print("-------------")
for i in range(len(pre)):
    if (pre[i] == 0):
        if (label_test[i] == 0):
            TN = TN + 1
        else:
            FN = FN + 1

    else:
        if (label_test[i] == 1):
            TP = TP + 1
        else:
            FP = FP + 1

print("TP = ",TP," FP = ",FP," TN = ",TN," FN = ",FN)
P = TP / (TP + FP)
R = TP / (TP + FN)
F1 = 2 * P * R / (P + R)
print("TPR = ", TP / (TP + FN), "FPR = ", FP / (TN + FP), "Precision", TP / (TP + FP),"F1-Score",F1)


end = time.time()
print("time is ",end - begin)

