import torch
import numpy as np
import time
import dnn
import LSTM
import newCNN
from torch.autograd import Variable

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device is: ", device)
Epoch = 2
batch_size = 8

data_path = "data/mawilab_6_10w.npy"
label_path = "data/mawilab_label_10w.npy"

print("data loading")
data = np.load(data_path)
labels = np.load(label_path)

train_size = 90000
test_size = 10000

begin = time.time()
data_train_all = data[0:train_size, :]
data_test = data[train_size:train_size + test_size, :]
label_train_all = labels[0:train_size]
label_test = labels[train_size:train_size + test_size]

print("loading done")
print(label_train_all.shape)
print(data_train_all.shape)

print("training")

# model = dnn.Model()
# model = LSTM.LSTMTagger()
model = newCNN.Model(90)
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.01)  # 0.00001,0.01
# print("sz: ", self.train_size, self.test_size)
train_batchs = train_size // batch_size
test_batchs = test_size // batch_size
data_test = torch.from_numpy(data_test).to(device)
label_test = torch.from_numpy(label_test).to(device)

model.train()
for epoch in range(Epoch):
    train_indices = np.random.permutation(train_size)
    data_train = torch.from_numpy(data_train_all[train_indices, :]).to(device)
    label_train = torch.from_numpy(label_train_all[train_indices]).to(device)
    print("train_batchs = ", train_batchs)
    for i in range(train_batchs):
        inputs = Variable(data_train[i * batch_size:min((i + 1) * batch_size, train_size), :],
                          requires_grad=False).view(-1, 1, data.shape[1])
        targets = Variable(label_train[i * batch_size:min((i + 1) * batch_size, train_size)],
                           requires_grad=False)

        num = min((i + 1) * batch_size, train_size) - i * batch_size + 1
        if num < batch_size:
            continue

        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = cost(outputs, targets)
        loss.backward()
        optimizer.step()

model.eval()

FP = 0
TN = 0
TP = 0
FN = 0
prediction = np.zeros(test_size, dtype=np.uint8)
# prob = np.zeros(test_size)

for i in range(test_batchs):
    inputs = Variable(data_test[i * batch_size:min((i + 1) * batch_size, test_size), :],
                      requires_grad=False).view(-1, 1, data.shape[1])
    targets = Variable(label_test[i * batch_size:min((i + 1) * batch_size, test_size)],
                       requires_grad=False)

    num = min((i + 1) * batch_size, test_size) - i * batch_size
    if num < batch_size:
        break

    outputs = model(inputs)

    pred = np.argmax(outputs.data.cpu().numpy(), axis=1)
    prediction[i * batch_size:min((i + 1) * batch_size, test_size)] = pred

    for j in range(len(pred)):
        if pred[j] == 0:
            if targets[j] == 0:
                TN = TN + 1
            else:
                FN += 1
        else:
            if targets[j] == 0:
                FP = FP + 1
            else:
                TP += 1

# print("shape: ", label_test.shape, prediction.shape)
# print("TN=", TN, "TP=", TP, "FP=", FP, "FN=", FN)
lb_cpu = label_test.data.cpu().numpy()
precision = precision_score(lb_cpu, prediction)
recall = recall_score(lb_cpu, prediction)
f1 = f1_score(lb_cpu, prediction)
if TN + FP > 0:
    fpr = FP / (TN + FP)
else:
    fpr = 0

print("testing")

pre = prediction

print("testing done")

print("f1 score = ", f1)
print("precision = ", precision)
print("recall = ", recall)
print("fpr = ", fpr)
print("time is ", time.time() - begin)
