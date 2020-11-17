import torch
from FeatureExtractor import FE
import numpy as np


# 0-1 正则化的类
class Normal01:
    def __init__(self, origin_dim):
        self.data_max = np.ones((origin_dim,)) * (-np.inf)
        self.data_min = np.ones((origin_dim,)) * np.inf

    # 0-1 正则化
    def norm(self, xx):
        for i in range(len(xx)):
            self.data_max[i] = max(self.data_max[i], xx[i])
            self.data_min[i] = min(self.data_min[i], xx[i])
        return ((xx - self.data_min) / (self.data_max - self.data_min + 1e-10)).astype(np.float32)


class DataManager:
    def __init__(self, data_file, label_file, save_data, save_label):
        print("data ", data_file, "init...")

        data_dim = 235
        self.norm01 = Normal01(data_dim)

        # 读取label文件
        f_label = open(label_file, "r", encoding='utf-8')
        self.label = []
        for row in f_label:
            x, y = row.strip().split(',')
            self.label.append(int(y))
        f_label.close()

        np.save("./data/" + save_label, np.array(self.label))

        self.fe = FE(data_file, np.inf)
        features = []

        x = 0
        while True:
            feature = self.fe.get_next_vector()

            if len(feature) == 0:
                break
            feature = self.norm01.norm(np.array(feature))
            features.append(feature)

            x += 1
            if x % 10000 == 0:
                print(x)

        ff = np.array(features)
        print("features is : ", ff.shape)
        np.save("./data/" + save_data, ff)

        print(data_file + " data init success")


d1 = DataManager("D:\\Dataset\\KITSUNE\\Mirai\\test.pcap.tsv", "E:\\dataset\\kitsune\\Mirai_labels.csv", "mirai.npy",
                 "mirai_label.npy")

d2 = DataManager("D:\\Dataset\\KITSUNE\\ARP_MitM\\test.pcap.tsv", "E:\\dataset\\kitsune\\ARP_MitM_labels.csv",
                 "arp.npy", "arp_label.npy")
d3 = DataManager("D:\\Dataset\\KITSUNE\\SSDP_Flood\\test.pcap.tsv", "E:\\dataset\\kitsune\\SSDP_Flood_labels.csv",
                 "ssdp_flood.npy", "ssdp_flood_label.npy")

d4 = DataManager("D:\\Dataset\\KITSUNE\\Fuzzing\\test.pcap.tsv", "E:\\dataset\\kitsune\\Fuzzing_labels.csv",
                 "fuzzing.npy", "fuzzing_label.npy")
