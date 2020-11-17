import numpy as np
import newCNN
import torch
from torch.autograd import Variable
from sklearn.metrics import f1_score, precision_score, recall_score

Epoch = 4
batch_size = 8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device is: ", device)


class CNNtest:
    def __init__(self):
        self.arp_data_all = np.load("D:\\Dataset\\kitsune_npy\\arp.npy")
        self.arp_label = np.load("D:\\Dataset\\kitsune_npy\\arp_label.npy").astype(np.longlong)
        self.mirai_data_all = np.load("D:\\Dataset\\kitsune_npy\\mirai.npy")
        self.mirai_label = np.load("D:\\Dataset\\kitsune_npy\\mirai_label.npy").astype(np.longlong)
        self.ssdp_data_all = np.load("D:\\Dataset\\kitsune_npy\\ssdp_flood.npy")
        self.ssdp_label = np.load("D:\\Dataset\\kitsune_npy\\ssdp_flood_label.npy").astype(np.longlong)
        self.fuzzing_data_all = np.load("D:\\Dataset\\kitsune_npy\\fuzzing.npy")
        self.fuzzing_label = np.load("D:\\Dataset\\kitsune_npy\\fuzzing_label.npy").astype(np.longlong)
        # print(self.arp_data_all.shape, self.mirai_data_all.shape, self.ssdp_data_all.shape, self.fuzzing_data_all.shape)

        self.F1s = np.load("./data/F1s.npy")
        self.Precisions = np.load("./data/Precisions.npy")
        self.Recalls = np.load("./data/Recalls.npy")
        self.FPRs = np.load("./data/FPRs.npy")

        self.mirai_train = 687724
        self.arp_train = 2253840
        self.ssdp_train = 3669539
        self.fuzzing_train = 2019725

        self.slides = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
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

    def train_test(self, train_num, data, label):
        model = newCNN.Model(data.shape[1]).to(device)
        cost = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.01)  # 0.00001,0.01

        test_size = data.shape[0] - train_num
        train_batchs = train_num // batch_size
        test_batchs = test_size // batch_size
        data_test = torch.from_numpy(data[train_num:]).to(device)
        label_test = torch.from_numpy(label[train_num:]).to(device)

        for epoch in range(Epoch):
            train_indices = np.random.permutation(train_num)
            data_train = torch.from_numpy(data[train_indices]).to(device)
            label_train = torch.from_numpy(label[train_indices]).to(device)

            model.train()

            for i in range(train_batchs):
                inputs = Variable(data_train[i * batch_size:min((i + 1) * batch_size, train_num), :],
                                  requires_grad=False).view(-1, 1, data.shape[1])
                targets = Variable(label_train[i * batch_size:min((i + 1) * batch_size, train_num)],
                                   requires_grad=False)

                num = min((i + 1) * batch_size, train_num) - i * batch_size + 1
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
                    if targets[j] == 0:
                        FP = FP + 1

        model.train()

        precision = precision_score(label_test, prediction)
        recall = recall_score(label_test, prediction)
        f1 = f1_score(label_test, prediction)
        if TN + FP > 0:
            fpr = FP / (TN + FP)
        else:
            fpr = 0
        return f1, precision, recall, fpr

    def _run(self, bin):  # 真运行
        print("running feature = ", self.bin2name(bin))
        sub = []
        x = bin
        for i in range(13):
            if x & 1 == 1:
                sub += self.slides[i]
            x = x >> 1

        if len(sub) <= 15:  # 单个一维特征或者无特征直接返回0
            self.F1s[bin][0] = self.F1s[bin][1] = self.F1s[bin][2] = self.F1s[bin][3] = 0
            return

        # mirai
        f1, precision, recall, fpr = self.train_test(self.mirai_train, self.mirai_data_all[:, sub], self.mirai_label)
        self.F1s[bin][0] = f1
        self.Precisions[bin][0] = precision
        self.Recalls[bin][0] = recall
        self.FPRs[bin][0] = fpr

        # arp_mitm
        f1, precision, recall, fpr = self.train_test(self.arp_train, self.arp_data_all[:, sub], self.arp_label)
        self.F1s[bin][1] = f1
        self.Precisions[bin][1] = precision
        self.Recalls[bin][1] = recall
        self.FPRs[bin][1] = fpr

        # SSDP_flood
        f1, precision, recall, fpr = self.train_test(self.ssdp_train, self.ssdp_data_all[:, sub], self.ssdp_label)
        self.F1s[bin][2] = f1
        self.Precisions[bin][2] = precision
        self.Recalls[bin][2] = recall
        self.FPRs[bin][2] = fpr

        # fuzzing
        f1, precision, recall, fpr = self.train_test(self.fuzzing_train, self.fuzzing_data_all[:, sub],
                                                     self.fuzzing_label)
        self.F1s[bin][3] = f1
        self.Precisions[bin][3] = precision
        self.Recalls[bin][3] = recall
        self.FPRs[bin][3] = fpr

        np.save("./data/F1s.npy", self.F1s)
        np.save("./data/Precisions.npy", self.Precisions)
        np.save("./data/Recalls.npy", self.Recalls)
        np.save("./data/FPRs.npy", self.FPRs)

    def run(self, bins):  # bin表示对应的子集的二进制, 如果没有被保存才运行
        ans = np.zeros(bins.shape[0])
        for i, bin in enumerate(bins):
            if np.isnan(self.F1s[bin][0]):
                self._run(bin)
            ans[i] = np.sum(self.F1s[bin]) / 4.0  # 取四个数据集的平均值
        return ans

    def bin2name(self, bin):
        name = "("
        if bin == 0:
            return "(null)"
        if bin & 1:
            name += "MIstat,"
        if (bin >> 1) & 1:
            name += "ARstat,"
        if (bin >> 2) & 1:
            name += "STstat_jit,"
        if (bin >> 3) & 1:
            name += "Scanstat,"
        if (bin >> 4) & 1:
            name += "HHstat_jit,"
        if (bin >> 5) & 1:
            name += "PPstat_jit,"
        if (bin >> 6) & 1:
            name += "HpHpstat,"
        if (bin >> 7) & 1:
            name += "SRstat,"
        if (bin >> 8) & 1:
            name += "SSLjit,"
        if (bin >> 9) & 1:
            name += "syn_jit,"
        if (bin >> 10) & 1:
            name += "Sjit,"
        if (bin >> 11) & 1:
            name += "Hstat,"
        if (bin >> 12) & 1:
            name += "HHstat"
        name += ")"
        return name
